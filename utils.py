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
# this is used for everything but bc-rnn
class MyData(Dataset):

    def __init__(self, data, loadname=None):
        if loadname:
            self.data = pickle.load(open(loadname, "rb"))
            self.data = torch.FloatTensor(self.data)
        else:
            self.data = torch.FloatTensor(data)
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


# import dataset with history for training
# this is used for bc-rnn
class MyDataHistory(Dataset):

    def __init__(self, loadname):
        self.data = pickle.load(open(loadname, "rb"))
        print("imported dataset of length:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return (torch.FloatTensor(self.data[idx][0]), 
                torch.FloatTensor(self.data[idx][1]))