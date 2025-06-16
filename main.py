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
STEAL = True
QUERY = True

if REQUEST_NEW_API:    
    model_stealer = ModelStealer(APIConfig.TOKEN)
    seed, port = model_stealer.request_new_api()

    print(f"New seed: {seed}, port: {port}.")


elif QUERY:
    dataset = torch.load("./data/ModelStealingPub.pt", weights_only=False)
    

elif STEAL:
    current_seed = "32454959"
    current_port = "9478"

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

