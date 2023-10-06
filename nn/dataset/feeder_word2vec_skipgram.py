import itertools
import math
import os
import pickle
import time
import traceback
import random
from collections import Counter

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


class SkipGramFeeder(torch.utils.data.Dataset):
    """
    load from disk
    """

    def __init__(self,
                 label_idx_filepath,
                 window_size=5,
                 neg_samples_per_center=5,
                 random_seed=None,
                 beat_divisor=8,
                 discard_hparam=0.0001,
                 neg_sampling_hparam=0.75,
                 take_first=None,
                 **kwargs,
                 ):
        """
        batch_size: number of subsequences in a batch. for rnn, one sample is a subsequence.
        """
        self.rand_inst = np.random.RandomState(random_seed)
        # self.save_dir = save_dir
        # self.mel_dir = os.path.join(save_dir, 'mel')
        # self.meta_dir = os.path.join(save_dir, 'meta')
        # self.label_dir = os.path.join(save_dir, 'label')
        # self.info_dir = os.path.join(save_dir, 'info')
        # self.label_idx_dir = os.path.join(save_dir, 'label_idx')
        self.label_idx_filepath = label_idx_filepath

        # self.all_beatmapids = [
        #     # str
        #     os.path.splitext(filename)[0]
        #     for filename in os.listdir(self.info_dir)
        # ]
        self.window_size = window_size
        self.neg_samples_per_center = neg_samples_per_center
        self.rand_inst = np.random.RandomState(random_seed)

        self.beat_divisor = beat_divisor
        self.label_idx_to_beat_label_seq = list(itertools.product(*[(0, 1, 2, 3) for _ in range(self.beat_divisor)]))
        self.beat_label_seq_to_label_idx = {seq: idx for idx, seq in enumerate(self.label_idx_to_beat_label_seq)}

        self.discard_hparam = discard_hparam
        self.neg_sampling_hparam = neg_sampling_hparam

        self.take_first = take_first

        self.div_subseq()

    def div_subseq(self):
        # for filename in os.listdir(self.label_idx_dir):
        #     filepath = os.path.join(self.label_idx_dir, filename)
        with open(self.label_idx_filepath, 'rb') as f:
            all_label_idx = pickle.load(f)
        print('totally records %d' % len(all_label_idx))
        if self.take_first is not None:
            all_label_idx = all_label_idx[:self.take_first]

        self.counter = Counter(itertools.chain.from_iterable(all_label_idx))
        total_num_label_idx = sum(self.counter.values())
        self.discard_prob = {k: max(1 - math.sqrt(self.discard_hparam / (v / total_num_label_idx)), 0)
                             for k, v in self.counter.items()}
        print('discard_prob')
        print(self.discard_prob)

        self.all_center = []
        self.all_context = []
        self.all_negative = []
        self.all_data = []
        self.all_label = []
        self.all_mask = []
        existent_label_idxs = set(self.counter.keys())
        neg_sampling_prob = {i: (self.counter[i] / total_num_label_idx) ** self.neg_sampling_hparam
                             for i in existent_label_idxs}
        for sample_idx, sample_label_idx in enumerate(tqdm(all_label_idx)):
            for window_start in range(len(sample_label_idx) - self.window_size):
                center_pos = window_start + self.window_size // 2
                center_label = sample_label_idx[center_pos]
                # subsample center label
                if self.rand_inst.rand() < self.discard_prob[center_label]:
                    continue
                # context len is always (window_size - 1)
                context = sample_label_idx[window_start:center_pos] + sample_label_idx[center_pos + 1:window_start + self.window_size]
                # subsample context
                mask = [0 if (
                            self.rand_inst.rand() < self.discard_prob[context_label]
                            or center_label == context_label
                        )
                        else 1 for context_label in context]
                if all([i == 0 for i in mask]):
                    continue
                candidates = existent_label_idxs.copy()
                candidates = candidates.difference(context + [center_label])
                p = np.array([neg_sampling_prob[i] for i in candidates])
                negative = self.rand_inst.choice(np.array(list(candidates)), len(context * self.neg_samples_per_center),
                                                 replace=True, p=p / np.sum(p))
                mask += [1] * len(negative)
                # if sample_idx == 0:
                #     print('center_label')
                #     print(center_label)
                #     print('context')
                #     print(context)
                #     print('negative')
                #     print(negative)
                #     print('mask')
                #     print(mask)
                context = np.array(context, dtype=int)

                self.all_center.append(np.array(center_label, dtype=int))
                self.all_context.append(context)
                self.all_negative.append(negative)
                self.all_data.append(np.concatenate([context, negative]))
                self.all_label.append(np.array([1] * len(context) + [0] * len(negative), dtype=int))
                self.all_mask.append(np.array(mask, dtype=int))
        print('total subseq %d' % len(self.all_center))

    def __len__(self):
        return len(self.all_center)

    def __getitem__(self, item):
        return self.all_center[item],\
               self.all_data[item],\
               self.all_label[item],\
               self.all_mask[item]
