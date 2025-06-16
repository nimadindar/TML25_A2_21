import torch
from torch.utils.data import Dataset

from PIL import Image
from typing import Tuple, List

from config import Augmentations


class TaskDataset(Dataset):

    def __init__(
            self,
            api_embeddings: torch.Tensor,):
        
        self.ids = []       
        self.imgs = []     
        self.labels = []         

        self.api_embeddings = api_embeddings
        self.transforms = Augmentations.AUGMENTATION_SET

    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index) -> List[torch.Tensor]:
        img = self.imgs[index]
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")
        views = [t(img) for t in self.transforms]
        target = self.api_embeddings[index]        
        return img, views, target
        