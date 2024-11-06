import json
import numpy as np
import torch
from itertools import cycle
from torch.utils.data import DataLoader


class SHDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = json.load(open(path, "r"))
        self.keys = list(self.data["sh"].keys())
        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = self.keys[idx]
        sh = np.array(self.data["sh"][key])
        sh = torch.tensor(sh, dtype=torch.float32)
        return sh


class SHDataset_byname(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = json.load(open(path, "r"))
        self.keys = list(self.data["sh"].keys())
        self.length = len(self.keys)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = self.keys[idx]
        sh = np.array(self.data["sh"][key])
        sh = torch.tensor(sh, dtype=torch.float32)
        return key, sh


def pick_random_k_shs(k, dataset):
    dataloader = cycle(DataLoader(dataset, batch_size=k, shuffle=True))
    shs = next(dataloader)
    return shs


def pick_random_k_shs_deterministic(k, dataset, seed):
    dataloader = cycle(DataLoader(dataset, batch_size=k, shuffle=False))
    torch.manual_seed(seed)
    shs = next(dataloader)
    return shs
