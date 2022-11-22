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

    def init(self):
        self.next_idx = self.batch_size
        self.cur_pos = [0] * self.batch_size
        self.cur_idx = list(range(self.batch_size))

    def __getitem__(self, index) -> T_co:
        return None

    def __iter__(self):
        """
        You should always use Dataloader(shuffle=False)
        whose batch size matches with self.batch_size

        will yield a batch!
        """
        data, label, clear_state = [], [], []
        for i, (sample_cur_pos, sample_cur_idx) in enumerate(zip(self.cur_pos, self.cur_idx)):
            if sample_cur_pos == 0:
                clear_state.append(True)
            else:
                clear_state.append(False)
            if sample_cur_idx < len(self.data):
                sample_data = self.data[sample_cur_idx][sample_cur_pos:sample_cur_pos + self.subseq_len]
            else:
                sample_data = [0.] * self.feature_dim
            data.append(torch.tensor(sample_data, dtype=torch.float))

            if not self.inference:
                if sample_cur_idx < len(self.data):
                    sample_label = self.label[sample_cur_idx][sample_cur_pos:sample_cur_pos + self.subseq_len]
                else:
                    sample_label = [0] * self.feature_dim
                label.append(torch.tensor(sample_label, dtype=torch.long))

            self.cur_pos[i] += self.subseq_len
            if sample_cur_pos >= self.data[sample_cur_idx].shape[1]:
                self.cur_pos[i] = 0
                self.cur_idx[i] = self.next_idx
                self.next_idx += 1

        if not self.inference:
            yield [data, label, clear_state]
        else:
            yield [data, clear_state]
