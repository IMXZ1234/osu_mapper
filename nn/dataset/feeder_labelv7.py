import math
import os
import pickle
import time
import traceback

import numpy as np
import torch
from torch.utils import data
import scipy
import collections


def normal_at(miu, sigma, length):
    return np.exp(-(np.abs(np.arange(length) - miu) / sigma) ** 2 / 2)


def gaussian_kernel_1d(sigma, size=9):
    if sigma == 0:
        kernel = np.zeros(size, dtype=float)
        kernel[size // 2] = 1
        return kernel
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
        filtered[i] = np.sum(img[i: i + kernel_size] * kernel)
    return filtered


def preprocess_label(label, level_coeff, rnd_bank=None):
    """
    only cursor signal
    -> x, y
    """
    if level_coeff == 0:
        return label
    if rnd_bank is None:
        label += level_coeff * np.random.randn(*label.shape) / 4096
    else:
        take_len = np.size(label)
        idx = np.random.randint(0, len(rnd_bank) - take_len)
        label += rnd_bank[idx:idx + take_len].reshape(label.shape)
    return label


def preprocess_label_filter(label, level_coeff, kernel):
    """
    only cursor signal
    -> x, y
    """
    # print('label fileter')
    if level_coeff == 0:
        return label
    kernel_size = kernel.shape[0]
    padded_label = np.pad(label, ((kernel_size // 2, kernel_size // 2), (0, 0)), mode='symmetric')
    filtered = np.stack([scipy.signal.convolve(label_i, kernel, mode='valid') for label_i in padded_label.T], axis=1)
    # filtered += level_coeff * np.random.randn(*label.shape) / 1024
    # filtered += level_coeff * np.random.randn(*label.shape) / 4096
    return filtered


def preprocess_embedding(embedding, level_coeff, rnd_bank=None):
    """
    hit signal embedding
    noise have probability ~90% to have a norm < 0.001
    embedding may move toward neighbour embeddings
    """
    if level_coeff == 0:
        return embedding
    if rnd_bank is None:
        rand_step = level_coeff * np.random.randn(*embedding.shape) / 1300
        gaussian = level_coeff * np.random.randn(*embedding.shape) / 2600
        former_step = np.concatenate([embedding[:1], embedding[:-1]])
        later_step = np.concatenate([embedding[1:], embedding[-1:]])
        abs_rand_step = np.abs(rand_step)
        dist = 1 - abs_rand_step
        adj_embedding = embedding * dist
        embedding = adj_embedding + abs_rand_step * np.where(
            rand_step >= 0,
            later_step,
            former_step
        ) + gaussian
    else:
        take_len = np.size(embedding)
        idx = np.random.randint(0, len(rnd_bank) - take_len)
        embedding += rnd_bank[idx:idx + take_len].reshape(embedding.shape)
    return embedding


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
                 # num snap
                 subseq_snaps,
                 # embedding_path,
                 use_random_iter=True,
                 take_occupied=False,
                 offset_meta_relative_to='aligned_audio',
                 pad=True,
                 # subseq_num for each batch
                 random_seed=None,
                 binary=False,
                 inference=False,
                 take_first=None,
                 take_first_subseq=None,
                 coeff_data_len=1,
                 coeff_label_len=1,
                 level_coeff=1,
                 embedding_level_coeff=1,
                 coord_level_coeff=None,
                 beat_divisor=8,
                 mel_frame_per_snap=8,
                 rnd_bank_size=None,
                 item='coord_type',
                 cache=128,
                 **kwargs,
                 ):
        """
        batch_size: number of subsequences in a batch. for rnn, one sample is a subsequence.
        # 8 mel frames per snap, 8 snaps per beat, 1 hit_signal embedding label per beat
        """
        self.rand_inst = np.random.RandomState(random_seed)
        self.save_dir = save_dir
        # self.embedding_filepath = embedding_path
        self.mel_dir = os.path.join(save_dir, 'mel')
        self.meta_dir = os.path.join(save_dir, 'meta')
        self.label_dir = os.path.join(save_dir, 'label')
        self.info_dir = os.path.join(save_dir, 'info')
        self.label_idx_dir = os.path.join(save_dir, 'label_idx')

        # with open(embedding_path, 'rb') as f:
        #     self.embedding = pickle.load(f)

        self.all_beatmapids = [
            # str
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.info_dir)
        ]
        # print(self.all_beatmapids)
        self.use_random_iter = use_random_iter
        self.take_occupied = take_occupied
        self.offset_meta_relative_to = offset_meta_relative_to
        self.take_first = take_first
        self.take_first_subseq = take_first_subseq

        self.beat_divisor = beat_divisor
        self.mel_frame_per_snap = mel_frame_per_snap

        self.subseq_snaps = subseq_snaps
        self.subseq_mel_frames = self.subseq_snaps * self.mel_frame_per_snap
        self.subseq_beats = self.subseq_snaps // self.beat_divisor
        self.coeff_data_len = coeff_data_len
        self.coeff_label_len = coeff_label_len
        self.level_coeff = level_coeff
        self.coord_level_coeff = coord_level_coeff
        # embedding noise is kept constant,
        # as a little noise will not interfere with decoding process,
        # but greatly stabilizes training
        self.embedding_level_coeff = embedding_level_coeff
        self.inference = inference
        self.rnd_bank_size = rnd_bank_size

        self.binary = binary
        self.pad = pad

        if self.take_first is not None:
            self.all_beatmapids = self.all_beatmapids[:self.take_first]
            print('taking first %d seq' % self.take_first)
        print('totally %d full seq' % len(self.all_beatmapids))
        self.info = {}

        self.meta_dict = {}
        self.label_dict = {}
        self.mel_dict = {}
        self.mel_dict_deque = collections.deque()
        self.max_mel_dict_len = cache

        for beatmapid in self.all_beatmapids:
            info_path = os.path.join(self.info_dir, '%s.pkl' % beatmapid)
            with open(info_path, 'rb') as f:
                self.info[beatmapid] = pickle.load(f)

            meta_path = os.path.join(self.meta_dir, '%s.pkl' % beatmapid)
            label_path = os.path.join(self.label_dir, '%s.pkl' % beatmapid)
            with open(meta_path, 'rb') as f:
                try:
                    self.meta_dict[beatmapid] = pickle.load(f)
                except Exception:
                    print('unable to read %s' % meta_path)
            with open(label_path, 'rb') as f:
                # n_snaps, 3(hit_object_type, x, y)
                try:
                    self.label_dict[beatmapid] = pickle.load(f)
                except Exception:
                    print('unable to read %s' % label_path)
        # print(self.info)

        # records where a subseq comes from
        # subseq_index: (sample_index, start, end, pad_len)
        self.subseq_dict = {}
        # records which subseqs belongs to a sample
        self.sample_subseq = {}
        self.divide_subseq()
        if self.take_first_subseq is not None:
            new_subseq_dict = {}
            keys = list(self.subseq_dict.keys())
            for i in range(self.take_first_subseq):
                k = keys[i]
                v = self.subseq_dict[k]
                new_subseq_dict[k] = v
            self.subseq_dict = new_subseq_dict
        print('dataset initialized, totally %d subseqs' % len(self.subseq_dict))

        self.set_noise_level(self.level_coeff)
        self.item = item
        # print(self.sample_subseq)
        # print(self.subseq_dict)
        # vis_data, vis_label, _ = self.__getitem__(0)
        # print(vis_label[:32, :])

    def set_noise_level(self, level_coeff):
        if self.rnd_bank_size is not None:
            self.rnd_bank = np.random.randn(self.rnd_bank_size)
            self.rnd_bank_label = self.rnd_bank * \
                                  ((
                                       self.coord_level_coeff if self.coord_level_coeff is not None else level_coeff) / 4096)
            self.rnd_bank_embedding = self.rnd_bank * \
                                      ((
                                           self.embedding_level_coeff if self.embedding_level_coeff is not None else level_coeff) / 1300)
            print('rnd_bank generated')
        else:
            self.rnd_bank = None
            self.rnd_bank_label = None
            self.rnd_bank_embedding = None
        self.kernel = gaussian_kernel_1d(sigma=level_coeff * 5)

    def divide_subseq(self):
        subseq_index = 0
        # print('len(self.info)')
        # print(len(self.info))
        for idx, (beatmapid, (total_mel_frames, beatmapset_id, first_occupied_snap, last_occupied_snap)) in enumerate(
                self.info.items()):
            if self.take_first is not None and idx > self.take_first:
                break
            self.sample_subseq[beatmapid] = []
            if self.take_occupied:
                # beat_offset relative to the first time position aligned to beats in the whole audio
                beat_offset = math.ceil(first_occupied_snap // self.beat_divisor)
                end_beat_offset = math.floor(last_occupied_snap // self.beat_divisor)
                total_beats = end_beat_offset - beat_offset
            else:
                total_beats = total_mel_frames // (self.mel_frame_per_snap * self.beat_divisor)
                beat_offset = 0
                end_beat_offset = beat_offset + total_beats
            if total_beats <= 0:
                print('beatmap too short')
                continue
            in_sample_subseq_num = total_beats // self.subseq_beats

            if self.pad:
                in_sample_subseq_num = math.ceil(in_sample_subseq_num)
            else:
                in_sample_subseq_num = int(in_sample_subseq_num)

            if not self.use_random_iter:
                # always cut from the beginning
                for in_sample_subseq_idx in range(in_sample_subseq_num):
                    start = in_sample_subseq_idx * self.subseq_beats
                    self.subseq_dict[subseq_index] = (beatmapid, beatmapset_id, start+beat_offset, beat_offset, end_beat_offset)
                    subseq_index += 1
                    self.sample_subseq[beatmapid].append(subseq_index)
            else:
                # cut from random position in sequence
                if self.pad:
                    # can start from arbitrary position
                    rand_end = total_beats
                else:
                    # avoid padding
                    rand_end = total_beats - self.subseq_beats
                    if rand_end <= 0:
                        continue
                start_list = self.rand_inst.randint(0, rand_end, size=[in_sample_subseq_num])
                for in_sample_subseq_idx in range(in_sample_subseq_num):
                    self.subseq_dict[subseq_index] = (beatmapid, beatmapset_id, start_list[in_sample_subseq_idx]+beat_offset, beat_offset, end_beat_offset)
                    subseq_index += 1
                    self.sample_subseq[beatmapid].append(subseq_index)

    def load_subseq(self, subseq_idx):
        # start is the beat index relative to the beat-aligned beginning of audio
        beatmapid, beatmapsetid, start, beat_offset, end_beat_offset = self.subseq_dict[subseq_idx]
        end = start + self.subseq_beats
        # real total sample beats
        type_coord = self.label_dict[beatmapid]
        sample_snaps = len(type_coord)
        sample_beats = sample_snaps // self.beat_divisor
        pad_beats = end - sample_beats

        type_coord = type_coord[start * self.beat_divisor: min(sample_snaps, end * self.beat_divisor)]

        if 'coord' in self.item:
            coord = type_coord[:, 1:].astype(np.float32)
            if pad_beats > 0:
                if not self.pad and (sample_beats >= (end - start)):
                    raise AssertionError
                #     print('label bad padding occur!')
                #     print(beatmapid, beatmapsetid, sample_beats, end, start)
                coord = np.pad(coord, ((0, pad_beats * self.beat_divisor), (0, 0)), mode='constant')
            coord_level_coeff = self.coord_level_coeff if self.coord_level_coeff is not None else self.level_coeff
            coord = preprocess_label_filter(coord, coord_level_coeff, self.kernel)

            coord = coord.T
            # print('coord.shape', coord.shape)
            # print('self.subseq_snaps', self.subseq_snaps)
            assert coord.shape[1] == self.subseq_snaps

        if 'type' in self.item:
            # L,
            ho_type = type_coord[:, 0].astype(int)
            if pad_beats > 0:
                if not self.pad and (sample_beats >= (end - start)):
                    raise AssertionError
                #     print('label bad padding occur!')
                #     print(beatmapid, beatmapsetid, sample_beats, end, start)
                ho_type = np.pad(ho_type, ((0, pad_beats * self.beat_divisor),), mode='constant')
            assert len(ho_type) == self.subseq_snaps

        if self.item == 'coord_type':
            return [ho_type, coord]
        elif self.item == 'coord':
            return [coord]
        elif self.item == 'type':
            return [ho_type]

    def __len__(self):
        return len(self.subseq_dict)

    def __getitem__(self, index):
        while True:
            try:
                label = self.load_subseq(index)
                # print(data.shape, label[0].shape, label[1].shape, meta_pred.shape)
                break
            except Exception:
                print('fail retrieve %d' % index)
                traceback.print_exc()
                index = (index + 1) % len(self.subseq_dict)
        return label

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


if __name__ == '__main__':
    train_dataset_arg = {
                            # 'save_dir': r'/home/data1/xiezheng/osu_mapper/preprocessed_v5',
                            'save_dir': r'/home/xiezheng/data/preprocessed_v5',
                            'embedding_path': '/home/data1/xiezheng/osu_mapper/result/word2vec_skipgramv1_0.1_constlr_dim_16_early_stop/embedding_center-1_-1_normed.pkl',
                            'subseq_snaps': 32*8,
                            'random_seed': 404,
                            'use_random_iter': True,
                            'take_first': 1024,
                            'pad': False,
                            'beat_divisor': 8,
                            'rnd_bank_size': None,
                            'level_coeff': 0.5,
                            'embedding_level_coeff': 0.5,
                            'coord_level_coeff': 0.5,
                            'item': 'coord_embedding',
                        }
    feeder = SubseqFeeder(
        **train_dataset_arg
    )
    # n_snaps_2
    label_mean = 0
    data_mean = 0
    from tqdm import tqdm

    for i in tqdm(range(len(feeder))):
        data, (label, embedding), meta_pred = feeder[i]
        label_mean += label
        data_mean += data
    data_mean = data_mean / len(feeder)
    label_mean = label_mean / len(feeder)
    save_dir = r'/home/data1/xiezheng/osu_mapper/vis'

    with open(os.path.join(save_dir, 'data_mean.pkl'), 'wb') as f:
        pickle.dump(data_mean, f)
    with open(os.path.join(save_dir, 'label_mean.pkl'), 'wb') as f:
        pickle.dump(label_mean, f)
    print(label_mean)

    import matplotlib.pyplot as plt


    plt.figure()
    plt.plot(np.arange(len(label_mean)), label_mean[:, 0])
    plt.savefig(os.path.join(save_dir, 'mean_x.png'))
    plt.figure()
    plt.plot(np.arange(len(label_mean)), label_mean[:, 1])
    plt.savefig(os.path.join(save_dir, 'mean_y.png'))

    plt.figure()
    plt.imshow(data_mean)
    plt.savefig(os.path.join(save_dir, 'mean_data.png'))
