import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sequence_length = 10

class DataGenerator(Dataset):
    def __init__(self, n=sequence_length, total=10000):
        self.n = n
        self.total = total

    def __getitem__(self, index):
        return torch.randperm(self.n)

    def __len__(self):
        return self.total

loader = DataLoader(DataGenerator(), batch_size=1)
