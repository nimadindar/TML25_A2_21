import os
from dotenv import load_dotenv
    
import torchvision.transforms as T

class APIConfig:
    load_dotenv()
    BASE_URL = "http://34.122.51.94:9090"
    TOKEN = os.getenv("TOKEN")
    IDX = 1 # This is used to keep track of the sequences of images queried through API.
    SEED = "32454959"
    PORT = "9478"


class TrainingConfig:
    ENCODER_NAME = "cifar10_resnet20"
    MODEL_IDX = 1
    SEED = 1234
    NUM_EPOCHS = 1
    BATCH_SIZE = 32
    NUM_AUGS = 4
    LR = 1e-3
    LAMBDA = 20
    
class Augmentations:

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    AUG_FLIP = T.Compose([
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    AUG_JITTER = T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    AUG_GRAYSCALE = T.Compose([
        T.RandomGrayscale(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    AUG_ROTATE = T.Compose([
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])

    AUGMENTATION_SET = [AUG_FLIP, AUG_JITTER, AUG_GRAYSCALE, AUG_ROTATE]