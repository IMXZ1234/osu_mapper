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
                 take_first=None,
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
        # else:
        #     self.label = [np.zeros([sample_data.shape[0]]) for sample_data in self.data]

        self.n_seq = len(self.data)
        self.data = [np.log(d + np.finfo(float).eps) for d in self.data]
        # print(self.data[0])
        # print('len(self.data[0])')
        # print(len(self.data[0]))
        # print(np.max([np.max(label[:, 1:], axis=0) for label in self.label], axis=0))
        # print(np.min([np.min(label[:, 1:], axis=0) for label in self.label], axis=0))

        self.seq_len = [
            self.data[i].shape[0]
            for i in range(self.n_seq)
        ]
        if self.inference:
            padded_data = []
            for d in self.data:
                length, feature_num = d.shape
                tail = length % self.subseq_len
                if tail == 0:
                    padded = d
                else:
                    padded = np.pad(d, ((0, self.subseq_len-tail), (0, 0)), mode='reflect')
                padded_data.append(padded)
            self.data = padded_data
        # print('self.seq_len')
        # print(self.seq_len)
        self.n_subseq = [
            self.seq_len[i] // self.subseq_len
            for i in range(self.n_seq)
        ]
        self.sample_div_pos = list(itertools.accumulate([0] + self.n_subseq))
        print('self.sample_div_pos')
        print(self.sample_div_pos)

        subseq_data_list = []
        for seq_data, n_subseq in zip(self.data, self.n_subseq):
            for j in range(n_subseq):
                start = j * self.subseq_len
                end = start + self.subseq_len
                subseq_data_list.append(seq_data[start:end])
        self.data = subseq_data_list

        if not self.inference:
            self.label = [process_label(label) for label in self.label]
            # print('label')
            # print(self.label[0])
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

    def cat_sample_labels(self, labels):
        all_sample_labels = []
        for i in range(len(self.sample_div_pos) - 1):
            all_sample_labels.append(torch.cat(labels[self.sample_div_pos[i]: self.sample_div_pos[i+1]]))
        return all_sample_labels
