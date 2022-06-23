import random

import numpy as np
import pickle
import torch
from torch.utils import data


class SubseqFeeder(torch.utils.data.Dataset):
    """
    Simplest feeder yielding (cond_data, label, index).
    Data may be either .pkl or .npy, style determined from path suffix.
    """

    def __init__(self,
                 data_path,
                 label_path,
                 subseq_len,
                 use_random_iter=True,
                 # subseq_num for each batch
                 random_seed=None,
                 binary=False,
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

        self.subseq_len = subseq_len

        self.binary = binary
        self.load_data()

    def load_data(self):
        # load label
        with open(self.label_path, 'rb') as f:
            self.label = pickle.load(f)

        if self.binary:
            for i in range(len(self.label)):
                self.label[i][np.where(self.label[i] != 0)[0]] = 1
        # load cond_data
        if self.data_path.endswith('.npy'):
            self.data = np.load(self.data_path)
        else:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)

        self.n_seq = len(self.data)

        self.seq_len = [
            self.data[i].shape[0]
            for i in range(self.n_seq)
        ]
        # print('self.seq_len')
        # print(self.seq_len)
        self.n_subseq = [
            self.seq_len[i] // self.subseq_len
            for i in range(self.n_seq)
        ]
        # print('self.n_subseq')
        # print(self.n_subseq)

        subseq_data_list, subseq_label_list = [], []
        for seq_data, seq_label, n_subseq in zip(self.data, self.label, self.n_subseq):
            for j in range(n_subseq):
                start = j * self.subseq_len
                end = start + self.subseq_len
                subseq_data_list.append(seq_data[start:end])
                subseq_label_list.append(seq_label[start:end])
        self.data = subseq_data_list
        self.label = subseq_label_list

        if self.use_random_iter:
            zipped = list(zip(self.data, self.label))
            random.shuffle(zipped)
            self.data, self.label = list(zip(*zipped))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index], index
