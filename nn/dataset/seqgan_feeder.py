import random
import math

import numpy as np
import pickle
import torch
from torch.utils import data
from torch.utils.data.dataset import T_co


class SeqGANFeeder(torch.utils.data.IterableDataset):
    """
    Simplest feeder yielding (cond_data, label, index).
    Data may be either .pkl or .npy, style determined from path suffix.
    """
    def __init__(self,
                 data_path,
                 label_path,
                 subseq_len,
                 # subseq_num for each batch
                 batch_size=8,
                 random_seed=None,
                 binary=False,
                 inference=False,
                 shuffle=True,
                 ):
        """
        batch_size: number of subsequences in a batch. for rnn, one sample is a subsequence.
        """
        self.random_inst = random.Random(random_seed)
        self.shuffle = shuffle

        self.data_path = data_path
        self.label_path = label_path

        self.subseq_len = subseq_len

        self.batch_size = batch_size
        self.binary = binary
        self.inference = inference
        self.load_data()

        self.init()

    def load_data(self):
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
            zipped = list(zip(self.data, self.label))
            self.random_inst.shuffle(zipped)
            self.data, self.label = list(zip(*zipped))
        else:
            self.random_inst.shuffle(self.data)

        self.n_seq = len(self.data)
        self.feature_dim = self.data[0].shape[1]
        self.label_dim = self.label[0].shape[1]

    def init(self):
        self.next_idx = self.batch_size
        self.cur_pos = 0
        self.cur_idx = list(range(min(self.batch_size, len(self.data))))
        self.batch_n_subseq = []
        self.cur_n_batch_subseq = 0

    def __getitem__(self, index) -> T_co:
        return None

    def __iter__(self):
        """
        You should always use Dataloader(shuffle=False)
        whose batch size matches with self.batch_size

        will yield a batch!
        """
        # other: (clear_state(boolean), valid_len(list of int with length of batch_size))
        data, label, valid_len = [], [], []
        clear_state = True
        for i, sample_cur_idx in enumerate(self.cur_idx):
            if len(self.data) >= self.cur_pos + self.subseq_len:
                sample_valid_len = self.subseq_len
                clear_state = False
            else:
                sample_valid_len = max(0, len(self.data) - self.cur_pos)

            sample_data = np.zeros([self.subseq_len, self.feature_dim])
            sample_data[:sample_valid_len] = self.data[sample_cur_idx][self.cur_pos:self.cur_pos + sample_valid_len]

            if not self.inference:
                sample_label = np.zeros([self.subseq_len, self.label_dim])
                sample_label[:sample_valid_len] = self.label[sample_cur_idx][self.cur_pos:self.cur_pos + sample_valid_len]

                label.append(torch.tensor(sample_label, dtype=torch.float))

            valid_len.append(sample_valid_len)
            data.append(torch.tensor(sample_data, dtype=torch.float))

        if not self.inference:
            yield [data, label, [clear_state, valid_len]]
        else:
            yield [data, [clear_state, valid_len]]

        self.cur_pos += self.subseq_len
        self.cur_n_batch_subseq += 1
        if clear_state:
            if self.next_idx > len(self.data):
                raise StopIteration
            self.cur_pos = 0
            self.cur_idx = list(range(self.next_idx, min(self.next_idx + self.batch_size, len(self.data))))
            self.next_idx += self.batch_size
            self.batch_n_subseq.append(self.cur_n_batch_subseq)
            self.cur_n_batch_subseq = 0

    def cat_sample_labels(self, labels):
        all_sample_labels = []
        sample_div_pos = np.cumsum(self.batch_n_subseq)
        for i in range(len(sample_div_pos) - 1):
            all_sample_labels.append(torch.cat(labels[sample_div_pos[i]: sample_div_pos[i+1]])[])
        return all_sample_labels