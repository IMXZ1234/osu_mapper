# sys
import numpy as np
import pickle
import torch
from torch.utils import data


class Feeder(torch.utils.data.Dataset):
    """
    Simplest feeder yielding (cond_data, label, index).
    Data may be either .pkl or .npy, style determined from path suffix.
    """

    def __init__(self,
                 data_path,
                 label_path,
                 **kwargs,
                 ):
        self.data_path = data_path
        self.label_path = label_path

        self.load_data()

    def load_data(self):
        # load label
        with open(self.label_path, 'rb') as f:
            self.label = pickle.load(f)

        # load cond_data
        if self.data_path.endswith('.npy'):
            self.data = np.load(self.data_path)
        else:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)
        # print('self.label.shape')
        # print(self.label.shape)
        # print('self.cond_data.shape')
        # print(self.cond_data.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get cond_data
        data_numpy = self.data[index]
        label = self.label[index]
        return data_numpy, label, index
