from dataset.dataset import TaskDataset, MergedDataset
from dataset.create_subset import get_random_subset

from encoders.cnn_encoder import StudentEncoder

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
STEAL = True
SUBMIT = False

if REQUEST_NEW_API:  
    model_stealer = ModelStealer(APIConfig.TOKEN)
    seed, port = model_stealer.request_new_api()
    print(f"New seed: {seed}, port: {port}.")

elif QUERY:
    if os.path.exists(f"./results/out{APIConfig.IDX}.pickle"):
        print(f"Representation file already exists for the given subset ID: {APIConfig.IDX}. \
              If you have obtained a new port/seed remove the existing file and try again.")
        sys.exit(0)
    else:
        dataset = torch.load("./data/ModelStealingPub.pt", weights_only=False)
        subset = get_random_subset(dataset, subset_index=APIConfig.IDX, seed=TrainingConfig.SEED)
        model_stealer = ModelStealer(APIConfig.TOKEN)

        image_ids = [dataset.ids[i] for i in subset.indices]
        images = [dataset.imgs[i] for i in subset.indices]

        representations = model_stealer.query_api(images, image_ids, APIConfig.IDX, APIConfig.PORT)

        print(f"The representation for the given subset with id {APIConfig.IDX} with length {len(representations)} has been fetched. \
                \n To view th output please refer to file saved in ./results/out{APIConfig.IDX}.pickle")

elif STEAL:
    dataset = torch.load("./data/ModelStealingPub.pt", weights_only=False)
    subset = get_random_subset(dataset, subset_index=APIConfig.IDX, subset_size=1000, seed=TrainingConfig.SEED)
    
    pickle_files = [
        'out0.pickle',
        'out1.pickle',
        'out2.pickle',
        'out3.pickle',
    ]

    out = {'ids': [], 'representations': []}
    for file in pickle_files:
        with open('./results/'+file, 'rb') as f:
            data = pickle.load(f)
            out['ids'].extend(data['ids'])
            out['representations'].extend(data['representations'])

    # Split into train (750) and validation (250)
    indices = np.random.permutation(1000)
    train_idx, val_idx = indices[:750], indices[750:]
    train_subset = torch.utils.data.Subset(subset, train_idx)
    val_subset = torch.utils.data.Subset(subset, val_idx)
    
    train_out = {'ids': [out['ids'][i] for i in train_idx], 'representations': [out['representations'][i] for i in train_idx]}
    val_out = {'ids': [out['ids'][i] for i in val_idx], 'representations': [out['representations'][i] for i in val_idx]}
    
    train_dataset = MergedDataset(train_subset, train_out)
    val_dataset = MergedDataset(val_subset, val_out)
    
    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.EVAL_BATCH_SIZE, shuffle=False)

    encoder = StudentEncoder(TrainingConfig.BACKBONE_TYPE)

    stolen_encoder = StolenEncoder(
        encoder, 
        TrainingConfig.LEARNING_RATE, 
        TrainingConfig.TRAINING_CYCLES, 
        TrainingConfig.DISTILLATION_WEIGHT,
        TrainingConfig.INVARIANCE_WEIGHT
    )

    if os.path.exists(f'./results/saved_models/stolen_model_{TrainingConfig.EXPERIMENT_ID}.pth'):
        print(f"The model with ID: {TrainingConfig.EXPERIMENT_ID} already exists. Loading the model to continue training...")
        encoder.load_state_dict(torch.load(f"./results/saved_models/stolen_model_{TrainingConfig.EXPERIMENT_ID}.pth")) 
        stolen_encoder.train(train_loader, val_loader, TrainingConfig.EXPERIMENT_ID + 1)

    print(f"Training the stolen encoder using subset id: {APIConfig.IDX}. The id for the model is {TrainingConfig.EXPERIMENT_ID}.")
    stolen_encoder.train(train_loader, val_loader, TrainingConfig.EXPERIMENT_ID)
    print("Training model finished successfully!")

elif SUBMIT:
    save_path = f'./results/saved_models/submission{APIConfig.SUB_IDX}'
    encoder = StudentEncoder(TrainingConfig.BACKBONE_TYPE)
    encoder.load_state_dict(torch.load(f"./results/saved_models/stolen_model_{TrainingConfig.EXPERIMENT_ID}.pth", map_location=torch.device('cpu')))
        
    torch.onnx.export(
        encoder,
        torch.randn(1, 3, 32, 32),
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