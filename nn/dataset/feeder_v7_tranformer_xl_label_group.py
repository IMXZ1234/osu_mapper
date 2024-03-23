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


class Feeder(torch.utils.data.Dataset):
    """
    load from disk
    """

    def __init__(self,
                 save_dir,
                 # num snap
                 subseq_snaps,
                 # embedding_path,
                 label_group_size=4,
                 use_random_iter=True,
                 take_occupied=False,
                 offset_meta_relative_to='aligned_audio',
                 pad=False,
                 # subseq_num for each batch
                 random_seed=None,
                 binary=False,
                 inference=False,
                 take_range=None,
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
                 bucket_size=128,  # num samples in a single bucket
                 take_beat_divisor=[2, 4, 8],  # num samples in a single bucket
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
        self.label_group_dir = os.path.join(save_dir, 'label_group_size%d' % label_group_size)
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
        self.take_range = take_range
        self.take_first_subseq = take_first_subseq

        self.beat_divisor = beat_divisor
        self.mel_frame_per_snap = mel_frame_per_snap
        self.label_group_size = label_group_size

        self.subseq_snaps = subseq_snaps
        self.subseq_label_groups = subseq_snaps // label_group_size
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

        if self.take_range is not None:
            self.all_beatmapids = self.all_beatmapids[self.take_range[0]:self.take_range[1]]
            print('taking range %s' % str(self.take_range))
        print('totally %d full seq' % len(self.all_beatmapids))
        self.info = {}

        self.meta_dict = {}
        self.label_group_dict = {}
        self.label_dict = {}
        self.mel_dict = {}
        self.mel_dict_deque = collections.deque()
        self.max_mel_dict_len = cache

        beatmapid_list = []
        num_mel_frames_list = []
        self.info_list = []
        self.beatmapid_to_beatmapsetid = {}

        for beatmapid in self.all_beatmapids:
            info_path = os.path.join(self.info_dir, '%s.pkl' % beatmapid)
            # former: total_mel_frames, beatmapset_id, first_occupied_snap, last_occupied_snap
            with open(info_path, 'rb') as f:
                beatmap_info = pickle.load(f)

            beatmap_beat_divisor = beatmap_info[-1]
            if beatmap_beat_divisor not in take_beat_divisor:
                continue
            self.info[beatmapid] = beatmap_info
            # (beatmap_id, num_beats)
            self.info_list.append((beatmapid, self.info[beatmapid][0] // (self.beat_divisor * self.mel_frame_per_snap)))
            self.beatmapid_to_beatmapsetid[beatmapid] = self.info[beatmapid][1]

            meta_path = os.path.join(self.meta_dir, '%s.pkl' % beatmapid)
            with open(meta_path, 'rb') as f:
                try:
                    self.meta_dict[beatmapid] = pickle.load(f)
                except Exception:
                    print('unable to read %s' % meta_path)
            label_path = os.path.join(self.label_dir, '%s.pkl' % beatmapid)
            with open(label_path, 'rb') as f:
                # n_snaps, 3(hit_object_type, x, y)
                try:
                    self.label_dict[beatmapid] = pickle.load(f)
                except Exception:
                    print('unable to read %s' % label_path)
            label_group_path = os.path.join(self.label_group_dir, '%s.pkl' % beatmapid)
            with open(label_group_path, 'rb') as f:
                # n_snaps, 3(hit_object_type, x, y)
                try:
                    self.label_group_dict[beatmapid] = pickle.load(f)
                except Exception:
                    print('unable to read %s' % label_group_path)
        sorted_info_list = list(sorted(self.info_list, key=lambda x: x[1]))
        self.sorted_beatmapid_list, self.sorted_num_beat_list = list(zip(*sorted_info_list))
        # print('self.sorted_num_beat_list', self.sorted_num_beat_list)
        split_point = np.arange(0, len(self.info_list)+1, bucket_size)
        if split_point[-1] != len(self.info_list):
            split_point = np.append(split_point, len(self.info_list))
        self.bucket_to_range = list(zip(split_point[:-1], split_point[1:]))

        self.index_to_bucket = []
        self.bucket_num_beat = []
        bucket_num_beat_diff = []
        for bucket_id in range(len(split_point) - 1):
            self.index_to_bucket.extend([bucket_id] * (split_point[bucket_id+1] - split_point[bucket_id]))

            max_num_beat = max(self.sorted_num_beat_list[split_point[bucket_id]: split_point[bucket_id + 1]])
            min_num_beat = min(self.sorted_num_beat_list[split_point[bucket_id]: split_point[bucket_id + 1]])
            bucket_num_beat_diff.append(max_num_beat - min_num_beat)
            # pad to multiples of subseq_beats
            self.bucket_num_beat.append(
                math.ceil(max_num_beat / self.subseq_beats) * self.subseq_beats
            )
        # print('bucket_num_beat_diff', bucket_num_beat_diff)

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

    def get_mel(self, beatmapsetid):
        if beatmapsetid in self.mel_dict:
            return self.mel_dict[beatmapsetid]

        mel_path = os.path.join(self.mel_dir, '%s.pkl' % beatmapsetid)
        # assemble feature
        with open(mel_path, 'rb') as f:
            mel_spec = pickle.load(f)
        mel_spec -= (np.mean(mel_spec, axis=0) + 1e-8)

        if len(self.mel_dict) >= self.max_mel_dict_len:
            self.mel_dict.pop(self.mel_dict_deque.popleft())
        self.mel_dict[beatmapsetid] = mel_spec
        self.mel_dict_deque.append(beatmapsetid)

        return mel_spec

    def load_beatmap(self, index):
        # start is the beat index relative to the beat-aligned beginning of audio
        # note beatmapid is str
        beatmapid = self.sorted_beatmapid_list[index]
        bucket_id = self.index_to_bucket[index]
        beatmapsetid = self.beatmapid_to_beatmapsetid[beatmapid]
        tgt_beats = self.bucket_num_beat[bucket_id]
        tgt_snaps = tgt_beats * self.beat_divisor
        tgt_steps = tgt_beats // self.subseq_beats

        # [mel_frame, n_mel=40]
        data = self.get_mel(beatmapsetid).astype(np.float32)
        n_mels = data.shape[1]
        assert (data.shape[0] % (self.mel_frame_per_snap * self.beat_divisor)) == 0, str(data.shape) + ' ' + str(tgt_snaps * self.mel_frame_per_snap)

        # real total sample beats
        sample_snaps = len(data) // self.mel_frame_per_snap
        sample_beats = sample_snaps // self.beat_divisor
        pad_beats = tgt_beats - sample_beats
        if pad_beats > 0:
            data = np.pad(data, ((0, pad_beats * self.beat_divisor * self.mel_frame_per_snap), (0, 0)), mode='constant')
        # denotes the relative position of this segment within
        # whole piece of music
        offset = np.arange(tgt_snaps * self.mel_frame_per_snap, dtype=np.float32) / (sample_snaps * self.mel_frame_per_snap)
        offset = offset[:, np.newaxis]
        data = np.concatenate([data, offset], axis=1)
        data = data.reshape([tgt_beats // self.subseq_beats, self.subseq_mel_frames, n_mels + 1])
        data = np.transpose(data, [2, 1, 0]).astype(np.float32)

        # *circle count, slider count, spinner count, *slider occupied, *spinner occupied
        # use meta from the whole sequence
        # meta = np.tile(np.array([star, cs])[np.newaxis, :], (len(data), 1))
        star, cs, ar, od, hp, bpm = self.meta_dict[beatmapid]
        # if self.offset_meta_relative_to == 'aligned_audio':
        #     offset_meta = start / (end_beat_offset - beat_offset)
        # elif self.offset_meta_relative_to == 'first_hitobject_beat':
        #     offset_meta = (start-beat_offset) / (end_beat_offset - beat_offset)
        # else:
        #     raise ValueError('unknown offset_meta_relative_to %s' % self.offset_meta_relative_to)
        meta = np.array([
            star / 10, (bpm-50) / 360
        ], dtype=np.float32)[:, np.newaxis, np.newaxis]
        meta = np.tile(meta, [1, self.subseq_mel_frames, tgt_steps])

        # meta is incorporated in data
        data = np.concatenate([data, meta], axis=0)
        # meta = np.array([star - 3.5]) / 5
        # print('meta', meta)
        # print('added indicator')
        # time_list.append(time.perf_counter_ns())

        # if not self.inference:
        # load label: (snap_type, x_pos_seq, y_pos_seq)
        type_coord = self.label_dict[beatmapid]
        # in range [-0.5, 0.5]
        coord = type_coord[:, 1:]
        coord = np.pad(coord, ((0, pad_beats * self.beat_divisor + 1), (0, 0)), mode='constant')
        coord_level_coeff = self.coord_level_coeff if self.coord_level_coeff is not None else self.level_coeff
        coord = preprocess_label_filter(coord, coord_level_coeff, self.kernel)

        # [n_snaps, 2]
        assert coord.shape[0] == tgt_snaps + 1
        out_coord = coord[1:, :]
        out_coord = np.transpose(out_coord.reshape([tgt_steps, self.subseq_snaps, 2]), [2, 1, 0])
        inp_coord = coord[:-1, :]
        inp_coord = np.transpose(inp_coord.reshape([tgt_steps, self.subseq_snaps, 2]), [2, 1, 0])

        # [n_snaps,]
        ho_type = self.label_group_dict[beatmapid].astype(int)
        ho_type = np.pad(ho_type, ((0, pad_beats * self.beat_divisor // self.label_group_size + 1),), mode='constant')
        # one_hot = np.eye(4)[ho_type][:, 1:].astype(np.float32)
        # embedding_level_coeff = self.embedding_level_coeff if self.embedding_level_coeff is not None else self.level_coeff
        # embedding = preprocess_embedding(one_hot, embedding_level_coeff, self.rnd_bank_embedding)

        assert len(ho_type) == tgt_snaps // self.label_group_size + 1
        out_ho_type = ho_type[1:]
        out_ho_type = out_ho_type.reshape([tgt_steps, self.subseq_label_groups]).T
        inp_ho_type = ho_type[:-1]
        inp_ho_type = inp_ho_type.reshape([tgt_steps, self.subseq_label_groups]).T

        # print('inp_ho_type', inp_ho_type)

        return data, inp_ho_type, inp_coord, out_ho_type, out_coord

    def __len__(self):
        # every beatmap is a single sample to be batched!
        return len(self.info)

    def __getitem__(self, index):
        while True:
            try:
                sample = self.load_beatmap(index)
                # print(data.shape, label[0].shape, label[1].shape, meta_pred.shape)
                break
            except Exception:
                print('fail retrieve %d' % index)
                traceback.print_exc()
                bucket_id = self.index_to_bucket[index]
                index_range = self.bucket_to_range[bucket_id]
                index += 1
                if index >= index_range[1]:
                    index = index_range[0]
        if self.inference:
            sample = sample[:3]
        return sample


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
    feeder = Feeder(
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
