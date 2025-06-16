import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from PIL import Image
from typing import Tuple, List

from config import Augmentations


class TaskDataset(Dataset):

    def __init__(self):
        
        self.ids = []       
        self.imgs = []     
        self.labels = []         

        # self.transforms = Augmentations.AUGMENTATION_SET

    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index) -> List[torch.Tensor]:
        id_ = self.ids[index]
        img = self.imgs[index]

        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        img_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=Augmentations.MEAN, std=Augmentations.STD)
        ])(img)

        views = [t(img) for t in Augmentations.AUGMENTATION_SET]
        return id_, img_tensor, views

    
class MergedDataset(Dataset):
    def __init__(self, subset, representations_dict):
        self.subset = subset  
        self.id_to_vector = dict(zip(representations_dict["ids"], representations_dict["representations"]))

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        id_, img, views = self.subset[index]  

        # Match representation using id
        if id_ not in self.id_to_vector:
            raise KeyError(f"ID {id_} not found in representations.")

        target = torch.tensor(self.id_to_vector[id_], dtype=torch.float32)

        return img, views, target 

        