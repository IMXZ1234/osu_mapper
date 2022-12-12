import random
import itertools

import numpy as np
import pickle
import torch
import os
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
    load from disk
    """

    def __init__(self,
                 save_dir,
                 subseq_len,
                 use_random_iter=True,
                 pad=False,
                 # subseq_num for each batch
                 random_seed=None,
                 binary=False,
                 inference=False,
                 take_first=None,
                 data_process_dim=128,  # do some preprocessing to data
                 snap_data_len=4,
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

        self.subseq_len = subseq_len
        self.snap_data_len = snap_data_len
        self.inference = inference
        self.data_process_dim = data_process_dim

        self.binary = binary
        self.pad = pad

        with open(self.info_path, 'rb') as f:
            self.info = pickle.load(f)
        print(self.info)

        # records where a subseq comes from
        # subseq_index: (sample_index, start, end, pad_len)
        self.subseq_dict = {}
        # records which subseqs belongs to a sample
        self.sample_subseq = {}
        self.divide_subseq()
        print(self.sample_subseq)
        print(self.subseq_dict)

    def divide_subseq(self):
        subseq_index = 0
        for sample_idx, sample_len in enumerate(self.info):
            if self.take_first is not None and sample_idx > self.take_first:
                break
            self.sample_subseq[sample_idx] = []
            in_sample_subseq_num = sample_len / self.subseq_len
            # calculate padding for the last subseq in this sample
            pad_len = 0
            if self.pad:
                in_sample_subseq_num = np.ceil(in_sample_subseq_num)
                pad_len = sample_len - (sample_len % self.subseq_len)
            else:
                in_sample_subseq_num = int(in_sample_subseq_num)

            for in_sample_subseq_idx in range(in_sample_subseq_num):
                start = in_sample_subseq_idx * self.subseq_len
                end = start + self.subseq_len
                # pad only the last subseq
                subseq_pad_len = 0 if in_sample_subseq_idx != in_sample_subseq_num - 1 else pad_len
                self.subseq_dict[subseq_index] = (sample_idx, start, end, subseq_pad_len)
                subseq_index += 1

                self.sample_subseq[sample_idx].append(subseq_index)

    def load_subseq(self, subseq_idx):
        sample_idx, start, end, pad_len = self.subseq_dict[subseq_idx]
        data_path = os.path.join(self.data_dir, '%d.pkl' % sample_idx)
        label_path = os.path.join(self.label_dir, '%d.pkl' % sample_idx)
        # load cond_data
        if data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        if pad_len > 0:
            data = np.pad(data, ((0, pad_len * self.snap_data_len), (0, 0)), mode='reflect')
        data = data[start * self.snap_data_len: end * self.snap_data_len]
        data = process_data(data, self.data_process_dim)

        if not self.inference:
            # load label
            with open(label_path, 'rb') as f:
                label = pickle.load(f)
            if pad_len > 0:
                label = np.pad(label, ((0, pad_len), (0, 0)), mode='reflect')
            label = label[start: end]
            label = process_label(label)
        else:
            label = None

        # if subseq_idx == 0:
        #     print('np.max(data), np.min(data)')
        #     print(np.max(data), np.min(data))
        return data, label

    def __len__(self):
        return len(self.subseq_dict)

    def __getitem__(self, index):
        data, label = self.load_subseq(index)
        if self.inference:
            return data, index
        else:
            return data, label, index

    def cat_sample_labels(self, labels):
        sample_label = {}
        for subseq_idx, label in enumerate(labels):
            sample_idx, start, end, pad_len = self.subseq_dict[subseq_idx]
            if pad_len > 0:
                # truncate padded portion
                label = label[:pad_len]
            if sample_idx not in sample_label:
                sample_label[sample_idx] = []
            sample_label[sample_idx].append((subseq_idx, label))
        all_sample_labels = []
        for sample_labels in sorted(sample_label.keys()):
            sample_labels = list(sorted(sample_labels, key=lambda x: x[0]))
            all_sample_labels.append(torch.cat(sample_labels, dim=0))
        return all_sample_labels
