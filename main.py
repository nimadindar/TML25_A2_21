from dataset.dataset import TaskDataset
from dataset.create_subset import get_random_subset

from encoders.cnn_encoder import CNNencoder

from stealing.query_api import ModelStealer
from stealing.train import StolenEncoder

from config import APIConfig, TrainingConfig, Augmentations

import numpy as np

import torch
from torch.utils.data import DataLoader

np.random.seed(TrainingConfig.SEED)
torch.cuda.manual_seed_all(TrainingConfig.SEED)


REQUEST_NEW_API = False
STEAL = False
QUERY = True

if REQUEST_NEW_API:    
    model_stealer = ModelStealer(APIConfig.TOKEN)
    seed, port = model_stealer.request_new_api()

    print(f"New seed: {seed}, port: {port}.")


elif QUERY:
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

    dataloader = DataLoader(subset, batch_size=TrainingConfig.BATCH_SIZE)

    encoder = CNNencoder(TrainingConfig.ENCODER_NAME)

    stolen_encoder = StolenEncoder(
        encoder, 
        TrainingConfig.LR, 
        TrainingConfig.NUM_EPOCHS, 
        TrainingConfig.LAMBDA)

    # stolen_encoder.train(dataloader, TrainingConfig.MODEL_IDX)

