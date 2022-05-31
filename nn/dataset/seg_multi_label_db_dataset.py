import bisect
import functools
import itertools
import os
import pickle
from collections import deque

import torch.nn.functional as F
from torch.utils.data import Dataset

from nn.dataset.dataset_util import hitobjects_to_label
from util import audio_util, beatmap_util
from preprocess import db


class SegMultiLabelDBDataset(Dataset):
    """
    Feed trainer with samples from OsuDB
    """

    def __init__(self, db_path, audio_dir,
                 table_name='MAIN',
                 beat_feature_frames=16384, snap_divisor=8, sample_beats=8, pad_beats=4,
                 multi_label=False,
                 inference=False):
        """
        pad_beats extra data is padded on both sides of one sample.
        Therefore, total beats for every sample is sample_beats + 2 * pad_beats,
        while total number of labels/(snaps whose hit object class are to be predicted)
        is sample_beats * snap_divisor.
        """
        super(SegMultiLabelDBDataset, self).__init__()
        # if inference, labels will not be yielded
        self.inference = inference
        # self.dataset_name = table_name
        database = db.OsuDB(db_path, connect=True)
        self.records = database.get_table_records(table_name)
        # self.beatmaps = [record[3] for record in self.records]
        self.beatmaps = [pickle.loads(record[3]) for record in self.records]
        self.audio_file_paths = [os.path.join(audio_dir, record[4]) for record in self.records]
        self.resample_rates_MHz = [record[5] / 1000000 for record in self.records]
        database.close()
        self.multi_label = multi_label
        # 8 is common snap divisor value
        self.snap_divisor = snap_divisor
        # adjust to multiples of snap_divisor
        self.beat_feature_frames = beat_feature_frames - beat_feature_frames % self.snap_divisor
        self.snap_feature_frames = self.beat_feature_frames // self.snap_divisor

        self.sample_beats = sample_beats
        self.pad_beats = pad_beats
        self.sample_beats_padded = self.sample_beats + self.pad_beats * 2

        # total beats for every sample
        self.sample_feature_frames = self.beat_feature_frames * self.sample_beats
        self.sample_feature_frames_padded = self.beat_feature_frames * self.sample_beats_padded
        self.pad_frames = self.beat_feature_frames * self.pad_beats

        self.sample_snaps = self.snap_divisor * self.sample_beats
        self.sample_snaps_pad = self.snap_divisor * self.pad_beats
        self.sample_snaps_padded = self.snap_divisor * self.sample_beats_padded
        # to calculate total number of samples
        # list of (bpm, start_time, end_time)
        self.audio_info_list = [(beatmap.bpm_min(),
                                 beatmap_util.get_first_hit_object_time_microseconds(beatmap),
                                 beatmap_util.get_last_hit_object_time_microseconds(beatmap))
                                for beatmap in self.beatmaps]
        self.beatmap_start_frame = [max(0, round(beatmap_info[1] * resample_rate_MHz))
                                    for beatmap_info, resample_rate_MHz
                                    in zip(self.audio_info_list, self.resample_rates_MHz)]
        self.groups = {}
        for idx, audio_file_path in enumerate(self.audio_file_paths):
            if audio_file_path not in self.groups:
                self.groups[audio_file_path] = []
            self.groups[audio_file_path].append(idx)
        self.audio_snap_num = [functools.partial(SegMultiLabelDBDataset.cal_snap_num,
                                                 snap_divisor=self.snap_divisor,
                                                 )(*beatmap_info)
                               for beatmap_info in self.audio_info_list]
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

        self.group_start_end_frame = {}
        for audio_file_path, group_beatmaps_idx in self.groups.items():
            group_start_frames = []
            group_end_frames = []
            for idx in group_beatmaps_idx:
                start_frame, end_frame = SegMultiLabelDBDataset.get_aligned_start_end_frame(
                    *self.audio_info_list[idx],
                    self.beat_feature_frames,
                    self.sample_beats,
                    self.audio_sample_num[idx]
                )
                group_start_frames.append(start_frame)
                group_end_frames.append(end_frame)
            self.group_start_end_frame[audio_file_path] = (min(group_start_frames), max(group_end_frames))

        if not self.inference:
            self.audio_snap_per_microsecond = [beatmap_info[0] * self.snap_divisor / 60000000
                                               for beatmap_info in self.audio_info_list]
            self.audio_label = [hitobjects_to_label(beatmap,
                                                    beatmap_info[1],  # first hit object time
                                                    snap_per_microsecond,
                                                    audio_snap_num,
                                                    self.sample_snaps,
                                                    self.multi_label)
                                for beatmap,
                                    beatmap_info,  # first hit object time
                                    snap_per_microsecond,
                                    audio_snap_num in
                                zip(self.beatmaps, self.audio_info_list, self.audio_snap_per_microsecond,
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
    def get_aligned_start_end_frame(bpm, start_time, end_time,
                                    beat_feature_frames=16384,
                                    sample_beats=8,
                                    audio_sample_num=None, ):
        resample_rate = round(beat_feature_frames * bpm / 60)  # * feature_frames_per_second
        sample_feature_frames = sample_beats * beat_feature_frames
        resample_rate_MHz = resample_rate / 1000000
        start_frame = max(0, round(start_time * resample_rate_MHz))
        end_frame = round(end_time * resample_rate_MHz)
        audio_calculated_sample_num = (end_frame - start_frame) // sample_feature_frames + 1
        if audio_sample_num is not None:
            if audio_calculated_sample_num < audio_sample_num:
                print('audio sample num larger than calculated!')
                # print('audio_calculated_sample_num')
                # print(audio_calculated_sample_num)
                # print('audio_sample_num')
                # print(audio_sample_num)
                audio_calculated_sample_num = audio_sample_num
        end_frame = audio_calculated_sample_num * sample_feature_frames + start_frame
        return start_frame, end_frame

    @staticmethod
    def preprocess(resampled_audio_data, start_frame, end_frame, pad_frames):
        """
        Pad/trim resampled audio data in database for train use.
        start_time, end_time in microsecond, should be aligned with snaps
        sample_rate in Hz
        Note we have two channels for audio data: audio_data.shape=[2, frame_num]
        """
        resampled_frame_num = resampled_audio_data.shape[1]
        # pad/trim at the start and end of the audio data
        pad_start_frame = pad_frames - start_frame
        pad_end_frame = pad_frames - (resampled_frame_num - end_frame)
        resampled_audio_data = resampled_audio_data[:,
                               -min(pad_start_frame, 0):min(pad_end_frame, 0) + resampled_frame_num]
        # F.pad pads the last dim if param `pad` contains only two values
        resampled_audio_data = F.pad(resampled_audio_data,
                                     (max(pad_start_frame, 0), max(pad_end_frame, 0)),
                                     mode='replicate')
        return resampled_audio_data

    # @staticmethod
    # def preprocess(resampled_audio_data,
    #                bpm, start_time, end_time,
    #                beat_feature_frames=32768,  # 512 * 64, 2**15
    #                sample_beats=8,
    #                audio_sample_num=None,
    #                pad_beats=4):
    #     """
    #     Pad/trim resampled audio data in database for train use.
    #     start_time, end_time in microsecond, should be aligned with snaps
    #     sample_rate in Hz
    #     Note we have two channels for audio data: audio_data.shape=[2, frame_num]
    #     """
    #     resampled_frame_num = resampled_audio_data.shape[1]
    #     resample_rate = round(beat_feature_frames * bpm / 60)  # * feature_frames_per_second
    #     resample_rate_MHz = resample_rate / 1000000
    #     start_time_frame = max(0, round(start_time * resample_rate_MHz))
    #     sample_feature_frames = sample_beats * beat_feature_frames
    #     # we ensure there are frames enough for an additional sample at the tail
    #     end_time_frame = round(end_time * resample_rate_MHz) + sample_feature_frames
    #
    #     # pad/trim at the start and end of the audio data
    #     pad_frames = pad_beats * beat_feature_frames
    #     pad_start_frame = pad_frames - start_time_frame
    #     pad_end_frame = pad_frames - (resampled_frame_num - end_time_frame)
    #     resampled_audio_data = resampled_audio_data[:,
    #                            -min(pad_start_frame, 0):min(pad_end_frame, 0) + resampled_frame_num]
    #     # F.pad pads the last dim if param `pad` contains only two values
    #     resampled_audio_data = F.pad(resampled_audio_data,
    #                                  (max(pad_start_frame, 0), max(pad_end_frame, 0)),
    #                                  mode='replicate')
    #     return resampled_audio_data

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
            preprocessed_audio_data, first_frame = cached
        else:
            # print('calculating')
            # print(audio_file_path)
            # this is resampled data
            audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
            preprocessed_audio_data = SegMultiLabelDBDataset.preprocess(audio_data,
                                                                        *(self.group_start_end_frame[audio_file_path]),
                                                                        self.pad_frames)
            first_frame = self.group_start_end_frame[audio_file_path][0]
            # if first_frame != self.beatmap_start_frame[audio_idx]:
            #     print('after pre first_frame')
            #     print(first_frame)
            #     print('after pre start_frame')
            #     print(self.beatmap_start_frame[audio_idx])
            #     print('bpm cal')
            #     print(self.beatmap_info_list[audio_idx][0])
            #     print('resample rate db')
            #     print(self.resample_rates_MHz[audio_idx])
            #     print('round(beat_feature_frames * bpm / 60)')
            #     print(round(self.beat_feature_frames * self.beatmap_info_list[audio_idx][0] / 60))
            self.cache_operation(audio_file_path, (preprocessed_audio_data, first_frame), op='put')
        # although different beatmaps in same beatmapset generally
        # shares one audio file,
        # these beatmap may require different preprocessing process before training
        # as start_time, end_time and audio_sample_num may be different.
        start_frame = self.beatmap_start_frame[audio_idx]
        if start_frame < first_frame:
            print('start_frame %d' % start_frame)
            print('first_frame %d' % first_frame)
            print('audio_file_path %s' % audio_file_path)
        speed_stars = self.speed_stars[audio_idx]
        # print('len(audio_label)')
        # print(len(audio_label))
        # print('preprocessed_audio_data.shape')
        # print(preprocessed_audio_data.shape)

        sample_feature_start_idx = start_frame - first_frame + sample_idx * self.sample_feature_frames
        sample_data = preprocessed_audio_data[:, sample_feature_start_idx:
                                                 sample_feature_start_idx + self.sample_feature_frames_padded]
        # print('data itv')
        # print('%d %d' % (sample_feature_start_idx, sample_feature_start_idx + self.sample_feature_frames_padded))
        sample_snap_start_idx = sample_idx * self.sample_snaps
        # print('label itv')
        # print('%d %d' % (sample_snap_start_idx, sample_snap_start_idx + self.sample_snaps))
        if self.inference:
            return [sample_data, speed_stars], index
        else:
            audio_label = self.audio_label[audio_idx]
            sample_label = audio_label[sample_snap_start_idx:sample_snap_start_idx + self.sample_snaps]
            # print('sample_data.shape')
            # print(sample_data.shape)
            # print('sample_label')
            # print(len(sample_label))
            return [sample_data, speed_stars], sample_label, index

# if __name__ == '__main__':
#     audio_file_path = r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\bgm\audio.mp3'
#     audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
#     print(sample_rate)
#     print(audio_data.shape)
#     feature_frames_per_beat = 512 * 16
#     bpm = 120
#     target_sample_rate = round(feature_frames_per_beat * bpm / 60)
#     resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, target_sample_rate)
#     print(resampled_audio_data.shape)
#     torchaudio.save(
#         r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\bgm\resampled.wav',
#         resampled_audio_data,
#         target_sample_rate,
#     )
