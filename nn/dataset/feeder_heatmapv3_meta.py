import math
import os
import pickle

import numpy as np
import torch
from torch.utils import data


def normal_at(miu, sigma, length):
    return np.exp(-(np.abs(np.arange(length) - miu) / sigma) ** 2 / 2)


def gaussian_kernel_1d(sigma, size=9):
    kernel = np.exp(-(np.abs(np.arange(size) - size // 2) / sigma) ** 2 / 2) / np.sqrt(2 * np.pi) / sigma
    kernel /= np.sum(kernel)
    return kernel


def filter_with_kernel_1d(img, kernel):
    kernel_size = kernel.shape[0]
    # only allow odd kernel size
    assert kernel_size % 2 == 1
    half_kernel_size = kernel_size // 2
    length = img.shape[0]
    filtered = np.zeros(img.shape)
    img = np.pad(img, (half_kernel_size, half_kernel_size))
    for i in range(length):
        filtered[i] = np.sum(img[i: i+kernel_size] * kernel)
    return filtered


def preprocess_label(label, level_coeff):
    """
    -> circle_heat_value, slider_heat_value, spinner_heat_value, x, y
    rescale hit signal to (0.1, 0.9) to avoid extreme value
    """
    label[:, :3] = 0.8 * label[:, :3] + 0.1
    label[:, :3] += level_coeff * np.random.randn(*label[:, :3].shape) / 64
    label[:, 3:] += level_coeff * np.random.randn(*label[:, 3:].shape) / 4096
    return label


def preprocess_mel(data):
    # to log scale
    data[...] = np.log10(data[...] + np.finfo(float).eps)
    return data


def to_data_feature(meta, mel_spec):
    (
        ms_per_beat, ho_density, occupied_proportion, snap_divisor,
    ) = meta[:4]
    # (
    #     # these are all floats
    #     diff_size, diff_overall, diff_approach, diff_drain, diff_aim, diff_speed, difficultyrating
    # ) = meta[4:]
    """
    mel_spec: [seq_len, n_mel]
    add meta into label, 64 + len(meta)
    
    do feature normalization
    """
    total_frames = mel_spec.shape[0]
    expanded_diff = [np.array(diff / 10. - 0.5, dtype=float).repeat(total_frames)[:, np.newaxis] for diff in meta[4:]]
    snap_divisor = np.array(snap_divisor, dtype=float).repeat(total_frames)
    sample_data = np.concatenate([mel_spec,
                                  ms_per_beat[:, np.newaxis] * 60 / 60000,
                                  ho_density[:, np.newaxis],
                                  occupied_proportion[:, np.newaxis],
                                  snap_divisor[:, np.newaxis] * 0.1,
                                 *expanded_diff], axis=1)
    return sample_data


def to_data_feature_dis(meta, mel_spec):
    # ms_per_beat, ho_density, occupied_proportion are time point related
    # (
    #     ms_per_beat, ho_density, occupied_proportion, snap_divisor,
    # ) = meta[:4]
    # (
    #     # these are all floats
    #     diff_size, diff_overall, diff_approach, diff_drain, diff_aim, diff_speed, difficultyrating
    # ) = meta[4:]
    """
    mel_spec: [seq_len, n_mel]
    add meta into label, 64 + len(meta)
    
    do feature normalization
    """
    total_frames = mel_spec.shape[0]
    expanded_diff = [np.array(diff / 10. - 0.5, dtype=float).repeat(total_frames)[:, np.newaxis] for diff in meta[4:8]]
    # snap_divisor = np.array(snap_divisor, dtype=float).repeat(total_frames)
    sample_data = np.concatenate([mel_spec,
                                  # ms_per_beat[:, np.newaxis] * 60 / 60000,
                                  # ho_density[:, np.newaxis],
                                  # occupied_proportion[:, np.newaxis],
                                  # snap_divisor[:, np.newaxis] * 0.1,
                                 *expanded_diff], axis=1)
    return sample_data


def preprocess_pred_meta(meta, mel_spec):
    """
    we let the discriminator predict metas which can be inferred from generated beatmap
    """
    (
        ms_per_beat, ho_density, occupied_proportion, snap_divisor,
    ) = meta[:4]
    # (
    #     # these are all floats
    #     diff_size, diff_overall, diff_approach, diff_drain, diff_aim, diff_speed, difficultyrating
    # ) = meta[4:]
    # return ms_per_beat[:, np.newaxis] * 60 / 60000, ho_density, occupied_proportion, snap_divisor[:, np.newaxis], diff_aim, diff_speed, difficultyrating
    total_frames = mel_spec.shape[0]
    expanded_diff = [np.array(diff / 10. - 0.5, dtype=float).repeat(total_frames)[:, np.newaxis] for diff in meta[8:]]
    snap_divisor = np.array(snap_divisor, dtype=float).repeat(total_frames)
    pred_meta = np.concatenate([ms_per_beat[:, np.newaxis] * 60 / 60000,
                                  ho_density[:, np.newaxis],
                                  occupied_proportion[:, np.newaxis],
                                  snap_divisor[:, np.newaxis] * 0.1,
                                 *expanded_diff], axis=1)
    return pred_meta


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
                 coeff_data_len=1,
                 coeff_label_len=1,
                 level_coeff=1,
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

        self.all_beatmapids = [
            # str
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.info_dir)
        ]
        self.use_random_iter = use_random_iter
        self.take_first = take_first

        self.subseq_len = subseq_len
        self.coeff_data_len = coeff_data_len
        self.coeff_label_len = coeff_label_len
        self.level_coeff = level_coeff
        self.inference = inference

        self.binary = binary
        self.pad = pad

        if self.take_first:
            self.all_beatmapids = self.all_beatmapids[16:16+self.take_first]
        self.info = {}
        for beatmapid in self.all_beatmapids:
            info_path = os.path.join(self.info_dir, '%s.pkl' % beatmapid)
            with open(info_path, 'rb') as f:
                self.info[beatmapid] = pickle.load(f)
        # print(self.info)

        # records where a subseq comes from
        # subseq_index: (sample_index, start, end, pad_len)
        self.subseq_dict = {}
        # records which subseqs belongs to a sample
        self.sample_subseq = {}
        self.divide_subseq()
        print('dataset initialized, totally %d subseqs' % len(self.subseq_dict))
        # print(self.sample_subseq)
        # print(self.subseq_dict)
        # vis_data, vis_label, _ = self.__getitem__(0)
        # print(vis_label[:32, :])

    def set_noise_level(self, level_coeff):
        self.level_coeff = level_coeff

    def divide_subseq(self):
        subseq_index = 0
        # print('len(self.info)')
        # print(len(self.info))
        for idx, (beatmapid, (sample_len, beatmapsetid, prelude_end_pos)) in enumerate(self.info.items()):
            # if self.take_first is not None and idx > self.take_first:
            #     break
            self.sample_subseq[beatmapid] = []
            in_sample_subseq_num = (sample_len - prelude_end_pos) / self.subseq_len

            if self.pad:
                in_sample_subseq_num = math.ceil(in_sample_subseq_num)
            else:
                in_sample_subseq_num = int(in_sample_subseq_num)

            if not self.use_random_iter:
                # always cut from the beginning
                for in_sample_subseq_idx in range(in_sample_subseq_num):
                    start = in_sample_subseq_idx * self.subseq_len
                    self.subseq_dict[subseq_index] = (beatmapid, start)
                    subseq_index += 1
                    self.sample_subseq[beatmapid].append(subseq_index)
            else:
                # cut from random position in sequence
                if self.pad:
                    # can start from arbitrary position
                    rand_end = sample_len
                else:
                    # avoid padding
                    rand_end = sample_len - self.subseq_len
                    if rand_end <= 0:
                        continue
                start_list = self.rand_inst.randint(prelude_end_pos, rand_end, size=[in_sample_subseq_num])
                for in_sample_subseq_idx in range(in_sample_subseq_num):
                    self.subseq_dict[subseq_index] = (beatmapid, beatmapsetid, start_list[in_sample_subseq_idx])
                    subseq_index += 1
                    self.sample_subseq[beatmapid].append(subseq_index)

    def load_subseq(self, subseq_idx):
        beatmapid, beatmapsetid, start = self.subseq_dict[subseq_idx]
        end = start + self.subseq_len

        mel_path = os.path.join(self.mel_dir, '%s.pkl' % beatmapsetid)
        meta_path = os.path.join(self.meta_dir, '%s.pkl' % beatmapid)
        label_path = os.path.join(self.label_dir, '%s.pkl' % beatmapid)

        # assemble feature
        with open(mel_path, 'rb') as f:
            mel_spec = pickle.load(f)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f).T
        data = mel_spec
        # mel_spec = preprocess_mel(mel_spec)
        # data = to_data_feature(meta, mel_spec)
        # data_dis = to_data_feature_dis(meta, mel_spec)
        # meta_pred = preprocess_pred_meta(meta, mel_spec)
        # data_dim, data_dis_dim, meta_pred_dim = data.shape[1], data_dis.shape[1], meta_pred.shape[1]

        sample_len = len(data) // self.coeff_data_len
        pad_len = end - sample_len
        # cat = np.concatenate([data, data_dis, meta_pred], axis=1)
        if pad_len > 0:
            data = np.pad(data, ((0, pad_len * self.coeff_data_len), (0, 0)), mode='constant')
            meta = np.pad(meta, ((0, pad_len * self.coeff_data_len), (0, 0)), mode='constant')
            if not self.pad:
                print('padding occur!')
        data = data[start * self.coeff_data_len: end * self.coeff_data_len]
        # *circle count, slider count, spinner count, *slider occupied, *spinner occupied
        meta = meta[start * self.coeff_data_len: end * self.coeff_data_len, np.array([0, 3, 4])]
        meta = np.mean(meta, axis=0)
        # print('meta', meta)

        if not self.inference:
            # load label
            with open(label_path, 'rb') as f:
                label = pickle.load(f)
            if pad_len > 0:
                if not self.pad:
                    print('padding occur!')
                label = np.pad(label, ((0, pad_len * self.coeff_label_len), (0, 0)), mode='constant')
            label = label[start * self.coeff_label_len: end * self.coeff_label_len]
            label = preprocess_label(label, self.level_coeff)
        else:
            label = None

        # if subseq_idx == 0:
        #     print('np.max(data), np.min(data)')
        #     print(np.max(data), np.min(data))
        return data, label, meta

    def __len__(self):
        return len(self.subseq_dict)

    def __getitem__(self, index):
        while True:
            try:
                data, label, meta_pred = self.load_subseq(index)
                break
            except Exception:
                index = (index + 1) % len(self.subseq_dict)
        if self.inference:
            return data, meta_pred
        else:
            return data, label, meta_pred

    def cat_sample_labels(self, labels):
        sample_label = {}
        for subseq_idx, label in enumerate(labels):
            sample_idx, start, end, pad_len = self.subseq_dict[subseq_idx]
            if pad_len > 0:
                # truncate padded portion
                label = label[:-pad_len]
            if sample_idx not in sample_label:
                sample_label[sample_idx] = []
            sample_label[sample_idx].append((subseq_idx, label))
        all_sample_labels = []
        for sample_idx in sorted(sample_label.keys()):
            idx_label_tuple = sample_label[sample_idx]
            idx_label_tuple = list(sorted(idx_label_tuple, key=lambda x: x[0]))
            subseq_idx_list, label_list = list(zip(*idx_label_tuple))
            all_sample_labels.append(torch.cat(label_list, dim=0))
        return all_sample_labels
