import random
import itertools

import numpy as np
import pickle
import torch
from torch.utils import data


class SubseqFeeder(torch.utils.data.Dataset):
    """
    Simplest feeder yielding (data, label, index).
    Data may be either .pkl or .npy, style determined from path suffix.
    """

    def __init__(self,
                 data_path,
                 label_path,
                 subseq_len,
                 use_random_iter=True,
                 flatten=False,
                 # subseq_num for each batch
                 random_seed=None,
                 binary=False,
                 inference=False,
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
        self.inference = inference

        self.binary = binary
        self.flatten = flatten
        self.load_data()

    def load_data(self):
        print(self.data_path)
        # load cond_data
        if self.data_path.endswith('.npy'):
            self.data = np.load(self.data_path)
        else:
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)

        if not self.inference:
            # load label
            with open(self.label_path, 'rb') as f:
                self.label = pickle.load(f)

            if self.binary:
                for i in range(len(self.label)):
                    self.label[i][np.where(self.label[i] != 0)[0]] = 1
        # else:
        #     self.label = [np.zeros([sample_data.shape[0]]) for sample_data in self.data]

        self.n_seq = len(self.data)

        self.seq_len = [
            self.data[i].shape[0]
            for i in range(self.n_seq)
        ]
        if self.inference:
            self.data = [
                np.concatenate(self.data[i], np.zeros())
                for i in range(self.n_seq)
            ]
        # print('self.seq_len')
        # print(self.seq_len)
        self.n_subseq = [
            self.seq_len[i] // self.subseq_len
            for i in range(self.n_seq)
        ]
        self.sample_div_pos = list(itertools.accumulate([0] + self.n_subseq))
        # print('self.n_subseq')
        # print(self.n_subseq)

        subseq_data_list = []
        for seq_data, n_subseq in zip(self.data, self.n_subseq):
            for j in range(n_subseq):
                start = j * self.subseq_len
                end = start + self.subseq_len
                subseq_data_list.append(seq_data[start:end])
        self.data = subseq_data_list

        if not self.inference:
            subseq_label_list = []
            for seq_label, n_subseq in zip(self.label, self.n_subseq):
                for j in range(n_subseq):
                    start = j * self.subseq_len
                    end = start + self.subseq_len
                    subseq_label_list.append(seq_label[start:end])
            self.label = subseq_label_list
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
            return self.data[index], index
        else:
            return self.data[index], self.label[index], index
