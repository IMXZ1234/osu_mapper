import math
import os
import pickle
import time
import traceback

import numpy as np
import torch
from torch.utils import data
import scipy


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
        filtered[i] = np.sum(img[i: i+kernel_size] * kernel)
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
        idx = np.random.randint(0, len(rnd_bank)-take_len)
        label += rnd_bank[idx:idx + take_len].reshape(label.shape)
    return label


def preprocess_label_filter(label, level_coeff, kernel):
    """
    only cursor signal
    -> x, y
    """
    if level_coeff == 0:
        return label
    filtered = np.stack([scipy.signal.convolve(label_i, kernel, mode='same') for label_i in label.T], axis=1)
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
        idx = np.random.randint(0, len(rnd_bank)-take_len)
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
                 embedding_path,
                 use_random_iter=True,
                 pad=False,
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
                 beat_divisor=8,
                 rnd_bank_size=None,
                 item='coord_embedding',
                 **kwargs,
                 ):
        """
        batch_size: number of subsequences in a batch. for rnn, one sample is a subsequence.
        # 16 mel frames per snap, 8 snaps per beat, 1 hit_signal embedding label per beat
        """
        self.rand_inst = np.random.RandomState(random_seed)
        self.save_dir = save_dir
        self.embedding_filepath = embedding_path
        self.mel_dir = os.path.join(save_dir, 'mel')
        self.meta_dir = os.path.join(save_dir, 'meta')
        self.label_dir = os.path.join(save_dir, 'label')
        self.info_dir = os.path.join(save_dir, 'info')
        self.label_idx_dir = os.path.join(save_dir, 'label_idx')

        with open(embedding_path, 'rb') as f:
            self.embedding = pickle.load(f)

        self.all_beatmapids = [
            # str
            os.path.splitext(filename)[0]
            for filename in os.listdir(self.info_dir)
        ]
        # print(self.all_beatmapids)
        self.use_random_iter = use_random_iter
        self.take_first = take_first
        self.take_first_subseq = take_first_subseq

        self.beat_divisor = beat_divisor

        self.subseq_snaps = subseq_snaps
        self.subseq_mel_frames = self.subseq_snaps * 16
        self.subseq_beats = self.subseq_snaps // self.beat_divisor
        self.coeff_data_len = coeff_data_len
        self.coeff_label_len = coeff_label_len
        self.level_coeff = level_coeff
        # embedding noise is kept constant,
        # as a little noise will not interfere with decoding process,
        # but greatly stabilizes training
        self.embedding_level_coeff = embedding_level_coeff
        self.inference = inference
        self.rnd_bank_size = rnd_bank_size

        self.binary = binary
        self.pad = pad

        if self.take_first:
            self.all_beatmapids = self.all_beatmapids[:self.take_first]
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
            self.rnd_bank_label = self.rnd_bank * (level_coeff / 4096)
            self.rnd_bank_embedding = self.rnd_bank * (self.embedding_level_coeff / 1300)
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
        for idx, (beatmapid, (total_mel_frames, beatmapset_id, first_occupied_snap, last_occupied_snap)) in enumerate(self.info.items()):
            if self.take_first is not None and idx > self.take_first:
                break
            self.sample_subseq[beatmapid] = []
            total_beats = total_mel_frames // (16 * self.beat_divisor)
            in_sample_subseq_num = total_beats // self.subseq_beats

            if self.pad:
                in_sample_subseq_num = math.ceil(in_sample_subseq_num)
            else:
                in_sample_subseq_num = int(in_sample_subseq_num)

            if not self.use_random_iter:
                # always cut from the beginning
                for in_sample_subseq_idx in range(in_sample_subseq_num):
                    start = in_sample_subseq_idx * self.subseq_beats
                    self.subseq_dict[subseq_index] = (beatmapid, start)
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
                    self.subseq_dict[subseq_index] = (beatmapid, beatmapset_id, start_list[in_sample_subseq_idx])
                    subseq_index += 1
                    self.sample_subseq[beatmapid].append(subseq_index)

    def load_subseq(self, subseq_idx):
        # time_list = []
        # print('start')
        # time_list.append(time.perf_counter_ns())
        # beatmapid, beatmapsetid, start, start_frame, end_frame = self.subseq_dict[subseq_idx]
        beatmapid, beatmapsetid, start = self.subseq_dict[subseq_idx]
        end = start + self.subseq_beats

        mel_path = os.path.join(self.mel_dir, '%s.pkl' % beatmapsetid)
        meta_path = os.path.join(self.meta_dir, '%s.pkl' % beatmapid)
        label_path = os.path.join(self.label_dir, '%s.pkl' % beatmapid)
        label_idx_path = os.path.join(self.label_idx_dir, '%s.pkl' % beatmapid)

        # assemble feature
        with open(mel_path, 'rb') as f:
            mel_spec = pickle.load(f)
        with open(meta_path, 'rb') as f:
            star, cs = pickle.load(f)
        # [mel_frame, n_mel=40]
        data = mel_spec
        # print('loaded')
        # time_list.append(time.perf_counter_ns())
        # mel_spec = preprocess_mel(mel_spec)
        # data = to_data_feature(meta, mel_spec)
        # data_dis = to_data_feature_dis(meta, mel_spec)
        # meta_pred = preprocess_pred_meta(meta, mel_spec)
        # data_dim, data_dis_dim, meta_pred_dim = data.shape[1], data_dis.shape[1], meta_pred.shape[1]

        # real total sample beats
        sample_snaps = len(data) // 16
        sample_beats = sample_snaps // self.beat_divisor
        pad_beats = end - sample_beats
        # cat = np.concatenate([data, data_dis, meta_pred], axis=1)
        data = data[start * self.beat_divisor * 16: min(len(data), end * self.beat_divisor * 16)]
        if pad_beats > 0:
            data = np.pad(data, ((0, pad_beats * self.beat_divisor * 16), (0, 0)), mode='constant')
            if (not self.pad) and (sample_beats >= (end - start)):
                raise AssertionError
            #     print('bad padding occur!')
            #     print(beatmapid, beatmapsetid, sample_beats, end, start)

        # print('padded')
        # time_list.append(time.perf_counter_ns())
        # populated_indicator = np.ones([self.subseq_snaps, 1])
        # if start < start_frame:
        #     populated_indicator[:start_frame-start, ...] = 0
        # if end > end_frame:
        #     populated_indicator[end_frame-end:, ...] = 0
        # data = np.concatenate([data, populated_indicator], axis=1)

        # *circle count, slider count, spinner count, *slider occupied, *spinner occupied
        # use meta from the whole sequence
        # -> [subseq_beats, 2]
        # meta = np.tile(np.array([star, cs])[np.newaxis, :], (len(data), 1))
        meta = np.array([star-3.5]) / 5
        # print('meta', meta)
        # print('added indicator')
        # time_list.append(time.perf_counter_ns())

        if not self.inference:
            # load label: (snap_type, x_pos_seq, y_pos_seq)
            if 'coord' in self.item:
                with open(label_path, 'rb') as f:
                    # n_snaps, 3
                    label = pickle.load(f)
                # to [-5, 0.5]
                # label = label[:, 1:] - 0.5
                label = (label[:, 1:] - 0.5) * 2
                # print('loaded label')
                # time_list.append(time.perf_counter_ns())
                label = label[start * self.beat_divisor: min(sample_snaps, end * self.beat_divisor)]
                if pad_beats > 0:
                    if not self.pad and (sample_beats >= (end - start)):
                        raise AssertionError
                    #     print('label bad padding occur!')
                    #     print(beatmapid, beatmapsetid, sample_beats, end, start)
                    label = np.pad(label, ((0, pad_beats * self.beat_divisor), (0, 0)), mode='constant')
                label = preprocess_label_filter(label, self.level_coeff, self.kernel)

            if 'embedding' in self.item:
                with open(label_idx_path, 'rb') as f:
                    # n_beats, 16
                    label_idx = pickle.load(f)
                label_idx = label_idx[start: min(sample_beats, end)]
                # print('padded label')
                # time_list.append(time.perf_counter_ns())
                # set mean norm of embedding to 1
                if pad_beats > 0:
                    if not self.pad and (sample_beats >= (end - start)):
                        raise AssertionError
                    label_idx = np.pad(label_idx, (0, pad_beats), mode='constant')
                embedding = self.embedding[label_idx]
                embedding = preprocess_embedding(embedding, self.embedding_level_coeff, self.rnd_bank_embedding)
                # embedding = self.embedding[label_idx] / 2.848476
            # print('preprocessed label')
            # time_list.append(time.perf_counter_ns())
        else:
            label = None
            embedding = None
        # print('finished')
        # time_list.append(time.perf_counter_ns())
        # print(np.diff(time_list))

        # if subseq_idx == 0:
        #     print('np.max(data), np.min(data)')
        #     print(np.max(data), np.min(data))
        if self.item == 'coord_embedding':
            return data, [label, embedding], meta
        elif self.item == 'coord':
            return data, label, meta
        elif self.item == 'embedding':
            return data, embedding, meta

    def __len__(self):
        return len(self.subseq_dict)

    def __getitem__(self, index):
        while True:
            try:
                data, label, meta_pred = self.load_subseq(index)
                # print(data.shape, label[0].shape, label[1].shape, meta_pred.shape)
                break
            except Exception:
                # print('fail retrieve %d' % index)
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
