from dataset.dataset import TaskDataset
from encoders.cnn_encoder import CNNencoder
from stealing.query_api import ModelStealer
from stealing.train import StolenEncoder

import random
import numpy as np
import torch

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)



REQUEST_NEW_API = False
STEAL = True


if REQUEST_NEW_API:
    
    seed, port = ModelStealer.request_new_api()