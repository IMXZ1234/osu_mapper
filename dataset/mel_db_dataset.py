import bisect
import functools
import itertools
import math
import os
import pickle
from collections import deque

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.dataset_util import hit_objects_to_label
from util import audio_util, beatmap_util
from preprocess import db


class MelDBDataset(Dataset):
    """
    Feed trainer with samples from OsuDB
    """

    def __init__(self, db_path, audio_dir,
                 table_name='MAIN',
                 snap_mel=4,
                 snap_offset=0,
                 snap_divisor=8,
                 sample_beats=8,
                 pad_beats=4,
                 multi_label=False,
                 inference=False):
        """
        pad_beats extra data is padded on both sides of one sample.
        Therefore, total beats for every sample is sample_beats + 2 * pad_beats,
        while total number of labels/(snaps whose hit object class are to be predicted)
        is sample_beats * snap_divisor.
        """
        super(MelDBDataset, self).__init__()
        # if inference, labels will not be yielded
        self.inference = inference
        # self.dataset_name = table_name
        database = db.OsuDB(db_path, connect=True)
        self.records = database.get_table_records(table_name)
        # self.beatmaps = [record[3] for record in self.records]
        self.beatmaps = [pickle.loads(record[3]) for record in self.records]
        self.audio_file_paths = [os.path.join(audio_dir, record[4]) for record in self.records]
        # self.resample_rates_MHz = [record[5] / 1000000 for record in self.records]
        self.crop_start_time = [record[6] for record in self.records]
        self.mel_frame_time = [record[8] for record in self.records]
        database.close()
        self.multi_label = multi_label
        # 8 is common snap divisor value
        self.snap_divisor = snap_divisor
        self.snap_offset = snap_offset
        # adjust to multiples of snap_divisor
        self.snap_mel = snap_mel
        self.beat_mel = self.snap_mel * self.snap_divisor

        self.sample_beats = sample_beats
        self.pad_beats = pad_beats
        self.sample_beats_padded = self.sample_beats + self.pad_beats * 2

        self.sample_snaps = self.snap_divisor * self.sample_beats
        self.sample_snaps_pad = self.snap_divisor * self.pad_beats
        self.sample_snaps_padded = self.snap_divisor * self.sample_beats_padded

        self.sample_mel = self.snap_mel * self.sample_snaps
        self.sample_mel_pad = self.snap_mel * self.sample_snaps_pad
        self.sample_mel_padded = self.snap_mel * self.sample_snaps_padded
        # to calculate total number of samples
        # list of (bpm, start_time, end_time)
        self.bpm_list = [beatmap.bpm_min() for beatmap in self.beatmaps]
        self.start_time_list = [beatmap_util.get_first_hit_object_time_microseconds(beatmap) for beatmap in self.beatmaps]
        self.end_time_list = [beatmap_util.get_last_hit_object_time_microseconds(beatmap) for beatmap in self.beatmaps]

        self.audio_snap_num = [MelDBDataset.cal_snap_num(bpm, start_time, end_time, self.snap_divisor)
                               for bpm, start_time, end_time in
                               zip(self.bpm_list, self.start_time_list, self.end_time_list)]
        if inference:
            # speed_stars should be calculated from hitobjects
            # however, during inference input beatmaps contains only a few metadata specified
            # by user, so we do not have hitobjects in beatmaps
            # here we use overall_difficulty to save user specified speed_stars
            self.speed_stars = [beatmap.overall_difficulty for beatmap in self.beatmaps]
        else:
            self.speed_stars = [beatmap_util.get_difficulty(beatmap) for beatmap in self.beatmaps]
        # self.audio_sample_num = [snap_num // self.sample_snaps for snap_num in self.audio_snap_num]
        self.audio_sample_num = [snap_num // self.sample_snaps for snap_num in self.audio_snap_num]
        self.accumulate_audio_sample_num = list(itertools.accumulate(self.audio_sample_num))

        self.audio_start_end_mel = [
            MelDBDataset.get_aligned_start_end_mel_frame(
                self.bpm_list[idx],
                self.start_time_list[idx],
                self.end_time_list[idx],
                self.crop_start_time[idx],
                self.snap_mel,
                self.mel_frame_time[idx],
                self.snap_offset,
                self.snap_divisor,
            ) for idx in range(len(self.beatmaps))
        ]

        # we preprocess(pad/trim) data for each audio once
        # samples data from different beatmaps in on beatmapset are slices of preprocessed audio data
        self.groups = {}
        for idx, audio_file_path in enumerate(self.audio_file_paths):
            if audio_file_path not in self.groups:
                self.groups[audio_file_path] = []
            self.groups[audio_file_path].append(idx)

        self.group_start_end_mel = {}
        for audio_file_path, group_beatmaps_idx in self.groups.items():
            grouped_start_end_mel = [self.audio_start_end_mel[idx] for idx in group_beatmaps_idx]
            start_mel_list, end_mel_list = list(zip(* grouped_start_end_mel))
            self.group_start_end_mel[audio_file_path] = (min(start_mel_list), max(end_mel_list))
        print(self.group_start_end_mel)

        if not self.inference:
            self.audio_snap_per_microsecond = [bpm * self.snap_divisor / 60000000
                                               for bpm in self.bpm_list]
            self.audio_label = [hit_objects_to_label(beatmap,
                                                     start_time,  # first hit object time
                                                     snap_per_microsecond,
                                                     audio_snap_num,
                                                     self.sample_snaps,
                                                     self.multi_label)
                                for beatmap,
                                    start_time,  # first hit object time
                                    snap_per_microsecond,
                                    audio_snap_num in
                                zip(self.beatmaps, self.start_time_list, self.audio_snap_per_microsecond,
                                    self.audio_snap_num)]
        self.cache_len = 2
        # we cache the audio data before final processing
        # we assume that the most time consuming part is I/O of audio data
        # just take up the place
        self.cache = {i: None for i in range(self.cache_len)}
        self.cache_order = deque(i for i in range(self.cache_len))

    def cache_operation(self, key, value=None, op='get'):
        ret_value = None
        if op == 'get':
            if key in self.cache:
                ret_value = self.cache[key]
        elif op == 'put':
            del self.cache[self.cache_order.popleft()]
            self.cache_order.append(key)
            self.cache[key] = value
        else:
            print('unknown op!')
        return ret_value

    @staticmethod
    def cal_snap_num(bpm, start_time, end_time, snap_divisor=8):
        """
        Determine how many samples there will be in one audio data.
        """
        snaps_per_microsecond = bpm * snap_divisor / 60000000
        if start_time < 0:
            # if first timing point is used to calculate valid start time
            # first timing point may be negative
            # usually we use first hit object to calculate valid start time
            print('get_valid_snap_interval: start time negative!')
            start_time = start_time % (1 / snaps_per_microsecond)
        # count the last snap
        total_snaps = round((end_time - start_time) * snaps_per_microsecond) + 1
        return total_snaps

    @staticmethod
    def time_to_relative_frame(time, start_time, sample_rate):
        return (time - start_time) * sample_rate

    @staticmethod
    def get_aligned_start_end_mel_frame(bpm,
                                        start_time,
                                        end_time,
                                        crop_start_time,
                                        snap_mel,
                                        mel_frame_time,
                                        snap_offset=0,
                                        snap_divisor=8):
        snap_time = 60000000 / snap_divisor / bpm
        offset_time = snap_offset * snap_time

        print('given mel_frame_time')
        print(mel_frame_time)
        print('calculated mel_frame_time')
        print(snap_time / snap_mel)
        mel_frame_time = snap_time / snap_mel

        start_mel_frame = round((start_time - crop_start_time - offset_time) / mel_frame_time)
        # align to snap
        start_mel_frame = round(start_mel_frame / snap_mel) * snap_mel
        end_mel_frame = round((end_time - crop_start_time - offset_time) / mel_frame_time)
        # align to snap
        end_mel_frame = math.ceil(end_mel_frame / snap_mel + 1) * snap_mel
        print('start_mel_frame before round')
        print((start_time - crop_start_time - offset_time) / mel_frame_time)
        print('start_mel_frame')
        print(start_mel_frame)
        print('end_mel_frame before round')
        print((end_time - crop_start_time - offset_time) / mel_frame_time)
        print('end_mel_frame')
        print(end_mel_frame)
        return start_mel_frame, end_mel_frame

    @staticmethod
    def preprocess(data, start_mel, end_mel, pad_mel):
        """
        Pad/trim resampled audio data in database for train use.
        start_time, end_time in microsecond, should be aligned with snaps
        sample_rate in Hz
        Note we have 3 channels for audio data: audio_data.shape=[2, freq, mel_frame]
        """
        # normalize intensity
        data = data / torch.mean(data)

        resampled_mel = data.shape[2]
        # pad/trim at the start and end of the audio data
        pad_start_mel = pad_mel - start_mel
        pad_end_mel = pad_mel - (resampled_mel - end_mel)
        data = data[:, :, -min(pad_start_mel, 0):min(pad_end_mel, 0) + resampled_mel]
        # F.pad pads the last dim if param `pad` contains only two values
        data = F.pad(data,
                     (max(pad_start_mel, 0), max(pad_end_mel, 0)),
                     mode='reflect')
        return data

    def __len__(self):
        return sum(self.audio_sample_num)

    def __getitem__(self, index):
        """
        Samples are aligned with beats
        """
        audio_idx = bisect.bisect(self.accumulate_audio_sample_num, index)
        sample_idx = index - self.accumulate_audio_sample_num[audio_idx - 1] if audio_idx > 0 else index
        # print('audio_idx')
        # print(audio_idx)
        # print('sample_idx')
        # print(sample_idx)
        audio_file_path = self.audio_file_paths[audio_idx]
        # print('audio_file_path')
        # print(audio_file_path)

        cached = self.cache_operation(audio_file_path, op='get')
        if cached is not None:
            # print('using cached')
            # print(audio_file_path)
            preprocessed_audio_data, group_start_mel = cached
        else:
            # print('calculating')
            # print(audio_file_path)
            # this is resampled data
            with open(audio_file_path, 'rb') as f:
                audio_data = pickle.load(f)
            preprocessed_audio_data = MelDBDataset.preprocess(audio_data,
                                                              *(self.group_start_end_mel[audio_file_path]),
                                                              self.sample_mel_pad)
            group_start_mel = self.group_start_end_mel[audio_file_path][0]
            # if group_start_mel != self.beatmap_start_frame[audio_idx]:
            #     print('after pre group_start_mel')
            #     print(group_start_mel)
            #     print('after pre audio_start_mel')
            #     print(self.beatmap_start_frame[audio_idx])
            #     print('bpm cal')
            #     print(self.beatmap_info_list[audio_idx][0])
            #     print('resample rate db')
            #     print(self.resample_rates_MHz[audio_idx])
            #     print('round(beat_feature_frames * bpm / 60)')
            #     print(round(self.beat_feature_frames * self.beatmap_info_list[audio_idx][0] / 60))
            self.cache_operation(audio_file_path, (preprocessed_audio_data, group_start_mel), op='put')
        # although different beatmaps in same beatmapset generally
        # shares one audio file,
        # these beatmap may require different preprocessing process before training
        # as start_time, end_time and audio_sample_num may be different.
        audio_start_mel = self.audio_start_end_mel[audio_idx][0]
        if audio_start_mel < group_start_mel:
            print('audio_start_mel %d' % audio_start_mel)
            print('group_start_mel %d' % group_start_mel)
            print('audio_file_path %s' % audio_file_path)
        speed_stars = self.speed_stars[audio_idx]
        bpm = self.bpm_list[audio_idx]
        # print('len(audio_label)')
        # print(len(audio_label))
        # print('preprocessed_audio_data.shape')
        # print(preprocessed_audio_data.shape)

        sample_start_mel = audio_start_mel - group_start_mel + sample_idx * self.sample_mel
        sample_data = preprocessed_audio_data[:, :, sample_start_mel:sample_start_mel + self.sample_mel_padded]
        # print('data itv')
        # print('%d %d' % (sample_start_mel, sample_start_mel + self.sample_mel_padded))
        sample_start_snap = sample_idx * self.sample_snaps
        if self.inference:
            return [sample_data, speed_stars, bpm], index
        else:
            audio_label = self.audio_label[audio_idx]
            sample_label = audio_label[sample_start_snap:sample_start_snap + self.sample_snaps]
            # print('label itv')
            # print('%d %d' % (sample_start_snap, sample_start_snap + self.sample_snaps))
            # print('sample_data.shape')
            # print(sample_data.shape)
            # print('sample_label')
            # print(len(sample_label))
            return [sample_data, speed_stars, bpm], sample_label, index
