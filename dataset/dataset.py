import torch
from torch.utils.data import Dataset

from typing import Tuple, List


class TaskDataset(Dataset):

    def __init__(
            self,
            api_embeddings: torch.Tensor,
            base_transform,
            num_augs: int = 4,):
        
        self.ids = []       
        self.imgs = []     
        self.labels = []         

        self.api_embeddings = api_embeddings
        self.base_transform = base_transform
        self.num_augs = num_augs

    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index) -> List[torch.Tensor]:
        img = self.imgs[index]
        views = [self.base_transform(img) for _ in range(self.num_augs)]
        target = self.api_embeddings[index]        
        return img, views, target
        