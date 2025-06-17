from dataset.dataset import TaskDataset, MergedDataset
from dataset.create_subset import get_random_subset

from encoders.cnn_encoder import CNNencoder

from stealing.query_api import ModelStealer
from stealing.train import StolenEncoder

from config import APIConfig, TrainingConfig, Augmentations

import numpy as np

import torch
from torch.utils.data import DataLoader
import onnxruntime as ort


import os
import sys
import pickle

np.random.seed(TrainingConfig.SEED)
torch.cuda.manual_seed_all(TrainingConfig.SEED)


REQUEST_NEW_API = False
QUERY = False
STEAL = False
SUBMIT = True


if REQUEST_NEW_API:  

    model_stealer = ModelStealer(APIConfig.TOKEN)
    seed, port = model_stealer.request_new_api()

    print(f"New seed: {seed}, port: {port}.")


elif QUERY:

    if os.path.exists(f"./results/out{APIConfig.IDX}"):
        print(f"Representation file already exists for the given subset ID: {APIConfig.IDX}. \
              If you have obtained a new port/seed remove the existing file and try again.")
        sys.exit(0)
    
    else:
        dataset = torch.load("./data/ModelStealingPub.pt", weights_only=False)
        subset = get_random_subset(dataset, subset_index=APIConfig.IDX, seed= TrainingConfig.SEED)
        model_stealer = ModelStealer(APIConfig.TOKEN)

        image_ids = [dataset.ids[i] for i in subset.indices]
        images = [dataset.imgs[i] for i in subset.indices]

        representations = model_stealer.query_api(images, image_ids, APIConfig.IDX, APIConfig.PORT)

        print(f"The representation for the given subset with id {APIConfig.IDX} with length {len(representations)} has been fetched. \
                \n To view th output please refer to file saved in ./results/out{APIConfig.IDX}.pickle")


elif STEAL:

    dataset = torch.load("./data/ModelStealingPub.pt", weights_only=False)
    subset = get_random_subset(dataset, subset_index=APIConfig.IDX, seed = TrainingConfig.SEED)
    
    try:
        with open(f'./results/out{APIConfig.IDX}.pickle', 'rb') as handle:
            out = pickle.load(handle)
    except FileNotFoundError:
        print(f"Representation file for subset index {APIConfig.IDX} not found. Please run the API querying step first.")
        exit(1)

    # To merge output representations with the subset of main dataset
    merged_dataset = MergedDataset(subset, out)

    dataloader = DataLoader(merged_dataset, batch_size=TrainingConfig.BATCH_SIZE)

    encoder = CNNencoder(TrainingConfig.ENCODER_NAME)

    stolen_encoder = StolenEncoder(
        encoder, 
        TrainingConfig.LR, 
        TrainingConfig.NUM_EPOCHS, 
        TrainingConfig.LAMBDA)

    print(f"Training the stolen encoder using subset id: {APIConfig.IDX}. The id for the model is {TrainingConfig.MODEL_IDX}.")
    stolen_encoder.train(dataloader, TrainingConfig.MODEL_IDX)
    print("Training model finished successfully!")

elif SUBMIT:
    
    save_path = f'./results/saved_models/submission{APIConfig.IDX}'

    encoder = CNNencoder(TrainingConfig.ENCODER_NAME)
    encoder.load_state_dict(torch.load(f"./results/saved_models/stolen_model_{TrainingConfig.MODEL_IDX}.pth"))
        
    torch.onnx.export(
        encoder,
        torch.randn(1,3,32,32),
        save_path,
        export_params=True,
        input_names=["x"],
    )

    with open(save_path, "rb") as f:
        encoder = f.read()
    try:
        stolen_model = ort.InferenceSession(encoder)
    except Exception as e:
        raise Exception(f"Invalid model, {e=}")
    try:
        out = stolen_model.run(
            None, {"x": np.random.randn(1, 3, 32, 32).astype(np.float32)}
        )[0][0]
    except Exception as e:
        raise Exception(f"Some issue with the input, {e=}")
    assert out.shape == (1024,), "Invalid output shape"

    model_stealer = ModelStealer(APIConfig.TOKEN)
    model_stealer.submit_model(APIConfig.SEED, save_path)

