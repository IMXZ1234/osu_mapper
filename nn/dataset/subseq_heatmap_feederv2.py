import os
import random
import itertools

import numpy as np
import pickle
import torch
from torch.utils import data
from torch.nn import functional as F


def process_label(label):
    """
    -> circle_heat_value, slider_heat_value, x, y
    """
    L, _ = label.shape
    # bounds in osu! beatmap editor
    x = (label[:, 1] + 180) / (691 + 180)
    y = (label[:, 2] + 82) / (407 + 82)
    # if snap is occupied by a hit_object,
    # noise's value should be almost always within -0.25~+0.25
    heat_value = np.random.randn(2 * L).reshape([L, 2]) / 8
    pos_circle = np.where(label[:, 0] == 1)[0]
    pos_slider = np.where(label[:, 0] == 2)[0]
    heat_value[pos_circle, np.zeros(len(pos_circle), dtype=int)] += 1
    heat_value[pos_slider, np.ones(len(pos_slider), dtype=int)] += 1
    return np.concatenate([heat_value, x[:, np.newaxis], y[:, np.newaxis]], axis=1)


def process_data(data, process_dim=128):
    data[:, :process_dim] = np.log10(data[:, :process_dim] + np.finfo(float).eps) / 15.
    return data


class SubseqFeeder(torch.utils.data.Dataset):
    """
    Simplest feeder yielding (data, label, index).
    Data may be either .pkl or .npy, style determined from path suffix.
    """

    def __init__(self,
                 save_dir,
                 use_random_iter=True,
                 # subseq_num for each batch
                 random_seed=None,
                 binary=False,
                 inference=False,
                 take_first=None,
                 data_process_dim=128,  # do some preprocessing to data
                 **kwargs,
                 ):
        """
        batch_size: number of subsequences in a batch. for rnn, one sample is a subsequence.
        """
        if random_seed is not None:
            random.seed(random_seed)
        self.save_dir = save_dir
        self.data_dir = os.path.join(self.save_dir, 'data')
        self.label_dir = os.path.join(self.save_dir, 'label')
        self.info_path = os.path.join(self.save_dir, 'info.pkl')
        self.use_random_iter = use_random_iter
        self.take_first = take_first

        self.inference = inference
        self.data_process_dim = data_process_dim

        self.binary = binary

        with open(self.info_path, 'rb') as f:
            self.info = pickle.load(f)
        print(self.info)

    def load_data(self, sample_idx):
        data_path = os.path.join(self.data_dir, '%d.pkl' % sample_idx)
        label_path = os.path.join(self.label_dir, '%d.pkl' % sample_idx)
        # load cond_data
        if data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        data = process_data(data, self.data_process_dim)

        if not self.inference:
            # load label
            with open(label_path, 'rb') as f:
                label = pickle.load(f)
            label = process_label(label)
        else:
            label = None

        return data, label

    def __len__(self):
        return len(self.info)

    def __getitem__(self, index):
        data, label = self.load_data(index)
        if self.inference:
            return data, index
        else:
            return data, label, index

