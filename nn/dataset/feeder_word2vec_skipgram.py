import math
import os
import pickle
import time
import traceback
import random

import numpy as np
import torch
from torch.utils import data


class SkipGramFeeder(torch.utils.data.IterableDataset):
    """
    load from disk
    """

    def __init__(self,
                 save_dir,
                 window_size=5,
                 num_neg_samples=7,
                 epoch_iter=999,
                 random_seed=None,
                 **kwargs,
                 ):
        """
        batch_size: number of subsequences in a batch. for rnn, one sample is a subsequence.
        """
        self.rand_inst = np.random.RandomState(random_seed)
        self.save_dir = save_dir
        self.mel_dir = os.path.join(save_dir, 'mel')
        self.meta_dir = os.path.join(save_dir, 'meta')
        self.label_dir = os.path.join(save_dir, 'label')
        self.info_dir = os.path.join(save_dir, 'info')
        self.label_idx_dir = os.path.join(save_dir, 'label_idx')

        self.all_beatmapids = [
            # str
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.info_dir)
        ]
        self.epoch_iter = epoch_iter
        self.num_neg_samples = num_neg_samples
        self.random_inst = random.Random(random_seed)

    def div_subseq(self):


    def __iter__(self):
        for _ in range(self.epoch_iter):
            try:
                data, label, meta_pred = self.load_subseq(index)
                break
            except Exception:
                index = (index + 1) % len(self.subseq_dict)
        if self.inference:
            return data, meta_pred
        else:
            return data, label, meta_pred
