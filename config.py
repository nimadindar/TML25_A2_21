import os
from dotenv import load_dotenv
import torchvision.transforms as T

class APIConfig:
    load_dotenv()
    BASE_URL = "http://34.122.51.94:9090"
    TOKEN = "34811541"
    IDX = 0
    SUB_IDX = 6
    SEED = "59427503"
    PORT = "9648"

class TrainingConfig:
    ENCODER_NAME = "resnet18"
    MODEL_IDX = 6
    SEED = 1234
    NUM_EPOCHS = 15
    BATCH_SIZE = 64
    VAL_BATCH_SIZE = 128
    LR = 3e-4
    LAMBDA_KD = 1.0
    LAMBDA_SIAM = 0.5

class Augmentations:
    MEAN = [0.2980, 0.2962, 0.2987]
    STD = [0.2886, 0.2875, 0.2889]

    AUG_TFM = T.Compose([
        T.RandomResizedCrop(32, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    AUGMENTATION_SET = [AUG_TFM] * 4