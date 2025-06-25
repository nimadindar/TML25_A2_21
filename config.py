import os
from dotenv import load_dotenv
import torchvision.transforms as T

class APIConfig:
    load_dotenv()
    BASE_URL = "http://34.122.51.94:9090"
    TOKEN = "34811541"
    IDX = 0
    SUB_IDX = 15
    SEED = "66577954"
    PORT = "9648"

class TrainingConfig:
    BACKBONE_TYPE = "resnet18"
    EXPERIMENT_ID = 15
    SEED = 1234
    epochs = 15
    TRAIN_BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 128
    LEARNING_RATE = 3e-4
    DISTILLATION_WEIGHT = 1.0
    INVARIANCE_WEIGHT = 0.5

class Augmentations:
    MEAN = [0.2980, 0.2962, 0.2987]
    STD = [0.2886, 0.2875, 0.2889]

    DATA_AUGMENTATION_PIPELINE = T.Compose([
        T.RandomResizedCrop(32, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=30),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        T.RandomApply([T.ColorJitter(0.5, 0.5, 0.3, 0.15)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
        T.RandomErasing(p=0.5, scale=(0.02, 0.1)),
    ])

    VIEW_GENERATION_SET = [DATA_AUGMENTATION_PIPELINE] * 4