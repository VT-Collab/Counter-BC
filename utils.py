import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset


# initialize the model weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


# import dataset for training
class MyData(Dataset):

    def __init__(self, loadname):
        self.data = pickle.load(open(loadname, "rb"))
        self.data = torch.FloatTensor(self.data)
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]