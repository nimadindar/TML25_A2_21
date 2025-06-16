import torch
from torch.utils.data import Dataset

from PIL import Image
from typing import Tuple, List

from config import Augmentations


class TaskDataset(Dataset):

    def __init__(self):
        
        self.ids = []       
        self.imgs = []     
        self.labels = []         

        self.transforms = Augmentations.AUGMENTATION_SET

    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index) -> List[torch.Tensor]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        views = [t(img) for t in self.transforms]
        return id_, img, views
        