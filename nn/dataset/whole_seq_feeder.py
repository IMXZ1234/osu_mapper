import random
import itertools

import numpy as np
import pickle
import torch
from torch.utils import data
from torch.nn import functional as F


def normalize_label(label):
    # bounds in osu! beatmap editor
    label[:, 1] = (label[:, 1] + 180) / (691 + 180)
    label[:, 2] = (label[:, 2] + 82) / (407 + 82)
    return label


class WholeSeqFeeder(torch.utils.data.Dataset):
    """
    Simplest feeder yielding (data, label, index).
    Data may be either .pkl or .npy, style determined from path suffix.
    """

    def __init__(self,
                 data_path,
                 label_path,
                 use_random_iter=True,
                 random_seed=None,
                 binary=False,
                 inference=False,
                 take_first=None,
                 ho_pos=False,
                 **kwargs,
                 ):
        """
        batch_size: number of subsequences in a batch. for rnn, one sample is a subsequence.
        """
        if random_seed is not None:
            random.seed(random_seed)
        self.data_path = data_path
        self.label_path = label_path
        self.use_random_iter = use_random_iter
        self.take_first = take_first
        self.ho_pos = ho_pos

        self.inference = inference

        self.binary = binary
        self.load_data()

    def load_data(self):
        print(self.data_path)
        # load cond_data
        if self.data_path.endswith('.npy'):
            self.data = np.load(self.data_path)
        else:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)

        if self.take_first is not None:
            self.data = self.data[:self.take_first]

        if not self.inference:
            # load label
            with open(self.label_path, 'rb') as f:
                self.label = pickle.load(f)

            if self.take_first is not None:
                self.label = self.label[:self.take_first]

            if self.binary:
                for i in range(len(self.label)):
                    self.label[i][np.where(self.label[i] != 0)[0]] = 1

        self.n_seq = len(self.data)

        self.seq_len = [
            self.data[i].shape[0]
            for i in range(self.n_seq)
        ]

        if not self.inference:
            if self.ho_pos:
                self.label = [normalize_label(label) for label in self.label]
        else:
            self.label = [None for _ in range(len(self.data))]

        if self.use_random_iter:
            zipped = list(zip(self.data, self.label))
            random.shuffle(zipped)
            self.data, self.label = list(zip(*zipped))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.inference:
            return self.data[index], index, self.seq_len[index]
        else:
            return self.data[index], self.label[index], self.seq_len[index]

    def cat_sample_labels(self, labels):
        all_sample_labels = []
        for i in range(len(self.sample_div_pos) - 1):
            all_sample_labels.append(torch.cat(labels[self.sample_div_pos[i]: self.sample_div_pos[i+1]]))
        return all_sample_labels
