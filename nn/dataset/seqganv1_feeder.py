import random

import numpy as np
import pickle
import torch
from torch.utils import data


class RNNv2Feeder(torch.utils.data.Dataset):
    """
    Simplest feeder yielding (cond_data, label, index).
    Data may be either .pkl or .npy, style determined from path suffix.
    """

    def __init__(self,
                 data_path,
                 label_path,
                 subseq_len,
                 use_random_iter=False,
                 # subseq_num for each batch
                 batch_size=8,
                 random_seed=None,
                 binary=False,
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

        self.batch_size = batch_size
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
        # print('cond_data.shape')
        # print([seq_data[0].shape for seq_data in self.cond_data])
        # print('label.shape')
        # print([seq_label.shape for seq_label in self.label])
        self.n_seq = len(self.data)
        # print('self.n_seq')
        # print(self.n_seq)

        self.seq_len = [
            self.data[i].shape[0]
            for i in range(self.n_seq)
        ]
        # print('self.seq_len')
        # print(self.seq_len)
        self.n_batch = [
            self.seq_len[i] // (self.subseq_len * self.batch_size)
            for i in range(self.n_seq)
        ]
        # print('self.n_batch')
        # print(self.n_batch)
        self.n_subseq = [
            self.n_batch[i] * self.batch_size
            for i in range(self.n_seq)
        ]
        # print('self.n_subseq')
        # print(self.n_subseq)
        self.redundant_len = [
            self.seq_len[i] - self.n_subseq[i] * self.subseq_len
            for i in range(self.n_seq)
        ]
        # print('self.redundant_len')
        # print(self.redundant_len)
        self.offset = [
            random.randint(0, self.redundant_len[i])
            if self.redundant_len[i] != 0 else 0
            for i in range(self.n_seq)
        ]
        # print('self.offset')
        # print(self.offset)

        self.out = []
        for i in range(len(self.data)):
            # clear state on the first batch of a sequence
            clear_state = True

            seq_data = self.data[i]
            seq_label = self.label[i]
            seq_offset = self.offset[i]
            seq_n_subseq = self.n_subseq[i]
            seq_n_batch = self.n_batch[i]

            seq_data = seq_data[seq_offset:]
            seq_label = seq_label[seq_offset:]

            init_idx = np.arange(0, seq_n_subseq * self.subseq_len, self.subseq_len)
            batch_init_idx = np.array_split(init_idx, seq_n_batch)
            # print('batch_init_idx')
            # print(batch_init_idx)

            for sample_init_idx in zip(*batch_init_idx):
                # yield a batch
                for idx in sample_init_idx:
                    self.out.append((
                        seq_data[idx:idx + self.subseq_len],
                        seq_label[idx:idx + self.subseq_len],
                        {'clear_state': clear_state}
                    ))
                clear_state = False

    def __len__(self):
        return np.sum(self.n_subseq)

    def __getitem__(self, index):
        """
        You should always use Dataloader(shuffle=False)
        whose batch size matches with self.batch_size
        """
        return self.out[index]
