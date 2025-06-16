import os
from dotenv import load_dotenv

class APIConfig:
    BASE_URL = "http://34.122.51.94:9090"
    load_dotenv()
    TOKEN = os.getenv("TOKEN")

class TrainingConfig:
    SEED = 1234
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    NUM_AUGS = 4
    LR = 1e-3
    