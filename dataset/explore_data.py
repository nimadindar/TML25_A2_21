import torch
import pickle
from dataset import TaskDataset

dataset = torch.load("./data/ModelStealingPub.pt", weights_only=False)
print(len(dataset))

print(dataset[0])

with open('./results/out.pickle', 'rb') as handle:
    out = pickle.load(handle)

print(len(out))

print(len(out[0]))