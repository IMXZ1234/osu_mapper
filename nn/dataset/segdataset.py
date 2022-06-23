import os
import pickle
from datetime import timedelta
import torch.nn.functional as F

import slider
import torch
import torchaudio
from torch.utils.data import Dataset

from util import audio_util, beatmap_util
from util.data import audio_osu_data
import itertools
import bisect
import functools
import math


def hit_objects_to_label(beatmap, aligned_start_time, snap_per_microsecond, total_snap_num, multi_label=False):
    """
    if multi_label:
    0: no hit object
    1: circle
    2: slider
    3: spinner
    4: hold note

    if not multi_label:
    0: no hit object
    1: has hit object
    """
    # print(audio_start_time_offset)
    label = [0 for _ in range(total_snap_num)]
    for hit_obj in beatmap.hit_objects():
        snap_idx = (hit_obj.time / timedelta(microseconds=1) - aligned_start_time) * snap_per_microsecond
        snap_idx = round(snap_idx)
        if isinstance(hit_obj, slider.beatmap.Circle):
            label[snap_idx] = 1
            continue
        end_snap_idx = (hit_obj.end_time / timedelta(microseconds=1) - aligned_start_time) * snap_per_microsecond
        if multi_label:
            if isinstance(hit_obj, slider.beatmap.Slider):
                label_value = 2
            elif isinstance(hit_obj, slider.beatmap.Spinner):
                label_value = 3
            else:
                # HoldNote
                label_value = 4
        else:
            label_value = 1
        end_snap_idx = round(end_snap_idx)
        for i in range(snap_idx, end_snap_idx + 1):
            label[i] = label_value
    return label


class SegDataset(Dataset):
    """
    Using index_file_path as reference.
    Load and convert mp3 file to wav on the fly to save disk space.
    Audio cond_data are cut into segments(samples), each used to predict the
    hit object class of the snap at the center of the segment.
    Same as in https://www.nicksypteras.com/blog/aisu.html
    """

    def __init__(self, index_file_path, beatmap_obj_file_path=None,
                 feature_frames_per_beat=32768, snap_divisor=8, beats_per_sample=9,
                 multi_label=False,
                 shuffle=False):
        super(SegDataset, self).__init__()
        self.audio_osu_data = audio_osu_data.AudioOsuData.from_path(index_file_path, beatmap_obj_file_path)
        if shuffle:
            self.audio_osu_data.use_random_iter()
        self.multi_label = multi_label
        # 8 is common snap divisor value
        self.snap_divisor = snap_divisor
        # adjust to multiples of snap_divisor
        self.feature_frames_per_beat = feature_frames_per_beat - feature_frames_per_beat % self.snap_divisor
        self.feature_frames_per_snap = self.feature_frames_per_beat // self.snap_divisor
        # adjust to odd number
        self.beats_per_sample = beats_per_sample + 1 - beats_per_sample % 2
        self.feature_frames_per_sample = self.feature_frames_per_beat * self.beats_per_sample
        self.half_feature_frames_per_sample = self.feature_frames_per_beat // 2
        self.snaps_per_sample = self.snap_divisor * self.beats_per_sample
        # to calculate total number of samples
        self.beatmap_info_list = [(beatmap.bpm_min(),
                                   beatmap_util.get_first_hit_object_time_microseconds(beatmap),
                                   beatmap_util.get_last_hit_object_time_microseconds(beatmap))
                                  for beatmap in self.audio_osu_data.beatmaps]
        # self.audio_valid_snap_interval_list = [SegDataset.get_valid_snap_interval(*beatmap_info)
        #                                        for beatmap_info in self.beatmap_info_list]
        # audio: audio cond_data of an audio file
        # sample: a segment of audio cond_data surrounding a snap
        # self.audio_sample_num = [valid_snap_interval[1] - valid_snap_interval[0]
        #                          for valid_snap_interval in self.audio_valid_snap_interval_list]
        self.audio_sample_num = [functools.partial(SegDataset.cal_snap_num,
                                                   snap_divisor=snap_divisor)(*beatmap_info)
                                 for beatmap_info in self.beatmap_info_list]
        # print('self.audio_sample_num')
        # print(self.audio_sample_num)
        self.accumulate_audio_sample_num = list(itertools.accumulate(self.audio_sample_num))

        self.last_preprocessed_audio = None
        self.last_audio_label = None
        self.last_speed_stars = None
        self.last_audio_idx = -1
        self.last_audio_file_path = ''

    def __len__(self):
        return sum(self.audio_sample_num)

    # @staticmethod
    # def get_valid_snap_interval(bpm, start_time, end_time, snap_divisor=8):
    #     """
    #     Determine which snaps should take part in the calculation of loss
    #     start_time should be time of the first expected hit object
    #     """
    #     snap_per_microsecond = bpm * snap_divisor / 60000000
    #     # [start_time, end_time] is the valid time interval during which hit objects exist.
    #     # outside this music can flow but no hit objects is assigned by the mapper.
    #     # such period of audio cond_data may act as a superior kind of padding cond_data,
    #     # as they contain helpful information.
    #     # get audio_start_time_offset, which should be on a snap
    #     if start_time < 0:
    #         # if first timing point is used to calculate valid start time
    #         # first timing point may be negative
    #         start_time = start_time % (1 / snap_per_microsecond)
    #     # aligned_start_time is the earliest time point which aligns with snaps
    #     # push the aligned_start_time as early as possible to keep the most
    #     # part of the useful audio cond_data
    #     aligned_start_time = start_time % (1 / snap_per_microsecond)
    #     valid_snap_interval = (round((start_time - aligned_start_time) * snap_per_microsecond),
    #                            round((end_time - aligned_start_time) * snap_per_microsecond) + 1)
    #     return valid_snap_interval

    # @staticmethod
    # def get_valid_snap_interval(bpm, start_time, end_time, snap_divisor=8):
    #     """
    #     Determine which snaps should take part in the calculation of loss
    #     start_time should be time of the first expected hit object
    #     """
    #     snap_per_microsecond = bpm * snap_divisor / 60000000
    #     # [start_time, end_time] is the valid time interval during which hit objects exist.
    #     # outside this music can flow but no hit objects is assigned by the mapper.
    #     # such period of audio cond_data may act as a superior kind of padding cond_data,
    #     # as they contain helpful information.
    #     # get audio_start_time_offset, which should be on a snap
    #     if start_time < 0:
    #         # if first timing point is used to calculate valid start time
    #         # first timing point may be negative
    #         print('get_valid_snap_interval: start time negative!')
    #         start_time = start_time % (1 / snap_per_microsecond)
    #     # aligned_start_time is the earliest time point which aligns with snaps
    #     # push the aligned_start_time as early as possible to keep the most
    #     # part of the useful audio cond_data
    #     aligned_start_time = start_time % (1 / snap_per_microsecond)
    #     valid_snap_interval = (round((start_time - aligned_start_time) * snap_per_microsecond),
    #                            round((end_time - aligned_start_time) * snap_per_microsecond) + 1)
    #     return valid_snap_interval

    @staticmethod
    def cal_snap_num(bpm, start_time, end_time, snap_divisor=8):
        """
        Determine which snaps should take part in the calculation of loss
        start_time should be time of the first expected hit object
        """
        snap_per_microsecond = bpm * snap_divisor / 60000000
        if start_time < 0:
            # if first timing point is used to calculate valid start time
            # first timing point may be negative
            print('get_valid_snap_interval: start time negative!')
            start_time = start_time % (1 / snap_per_microsecond)
        total_snap_num = (end_time - start_time) * snap_per_microsecond
        # print('total_snap_num - 1')
        # print(total_snap_num)
        total_snap_num = round(total_snap_num) + 1
        return total_snap_num

    @staticmethod
    def preprocess(audio_data, sample_rate,
                   bpm, start_time, end_time,
                   feature_frames_per_beat=32768,  # 512 * 64, 2**15
                   beats_per_sample=9, ):
        """
        Resample the audio cond_data to expand feature_frames_per_beat to target value
        Generally we have bpm of about 120 and sample rate of 44100Hz,
        which leads to about 22100 frames per beat.
        start_time, end_time in microsecond, should be aligned with snaps
        sample_rate in Hz
        Note we have two channels for audio cond_data: audio_data.shape=[2, frame_num]
        """
        # print('bpm, start_time, end_time')
        # print(bpm, start_time, end_time)
        resample_rate = round(feature_frames_per_beat * bpm / 60)  # * feature_frames_per_second
        # print('resample_rate')
        # print(resample_rate)
        # alter length of audio with respective to bpm to ensure
        # different audio have same number of frames for each beat
        resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, resample_rate)
        # print('resampled_audio_data.shape')
        # print(resampled_audio_data.shape)
        resampled_frame_num = resampled_audio_data.shape[1]
        resample_rate_MHz = resample_rate / 1000000
        start_time_frame = max(0, round(start_time * resample_rate_MHz))
        end_time_frame = min(resampled_frame_num - 1, round(end_time * resample_rate_MHz))
        half_frames_per_sample = beats_per_sample * feature_frames_per_beat // 2
        pad_start_frame = half_frames_per_sample - start_time_frame
        pad_end_frame = half_frames_per_sample - (resampled_frame_num - 1 - end_time_frame)
        # print('start_time_frame')
        # print(start_time_frame)
        # print('end_time_frame')
        # print(end_time_frame)
        # print('start_time_frame - end_time_frame')
        # print(start_time_frame - end_time_frame)
        # print('snap num resample ')
        # print((start_time_frame - end_time_frame) * 8 / feature_frames_per_beat)
        resampled_audio_data = resampled_audio_data[:, -min(pad_start_frame, 0):
                                                       -min(pad_end_frame, 0) + resampled_frame_num]
        # F.pad pads the last dim if param `pad` contains only two values
        resampled_audio_data = F.pad(resampled_audio_data,
                                     (max(pad_start_frame, 0), max(pad_end_frame, 0)),
                                     mode='replicate')
        # print('resampled_audio_data.shape')
        # print(resampled_audio_data.shape)
        return resampled_audio_data

    def __getitem__(self, index):
        audio_idx = bisect.bisect(self.accumulate_audio_sample_num, index)
        sample_idx = index - self.accumulate_audio_sample_num[audio_idx - 1] if audio_idx > 0 else index
        # print('audio_idx')
        # print(audio_idx)
        # print('sample_idx')
        # print(sample_idx)

        beatmap = self.audio_osu_data.beatmaps[audio_idx]
        audio_file_path = self.audio_osu_data.audio_osu_list[audio_idx][0]

        if self.last_audio_idx == audio_idx:
            # print('using cached')
            preprocessed_audio_data = self.last_preprocessed_audio
            audio_label = self.last_audio_label
            speed_stars = self.last_speed_stars
        else:
            # print('calculating')
            audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
            beatmap_info = self.beatmap_info_list[audio_idx]
            # print('beatmap_info')
            # print(beatmap_info)
            preprocessed_audio_data = SegDataset.preprocess(audio_data, sample_rate,
                                                            *beatmap_info,
                                                            self.feature_frames_per_beat,
                                                            self.beats_per_sample)
            self.last_preprocessed_audio = preprocessed_audio_data

            snap_per_microsecond = self.beatmap_info_list[audio_idx][0] * self.snap_divisor / 60000000
            # print('snap_per_microsecond')
            # print(snap_per_microsecond)
            audio_label = hit_objects_to_label(beatmap,
                                               beatmap_info[1],  # first hit object time
                                               snap_per_microsecond,
                                               self.audio_sample_num[audio_idx],
                                               self.multi_label)
            self.last_audio_label = audio_label
            # print('len(audio_label)')
            # print(len(audio_label))

            speed_stars = beatmap_util.get_difficulty(beatmap)
            self.last_speed_stars = speed_stars

            self.last_audio_idx = audio_idx

        sample_feature_start_idx = sample_idx * self.feature_frames_per_snap
        sample_preprocessed_audio_data = preprocessed_audio_data[:, sample_feature_start_idx:
                                                                    sample_feature_start_idx + self.feature_frames_per_sample]
        return [sample_preprocessed_audio_data, speed_stars], audio_label[sample_idx], index


if __name__ == '__main__':
    audio_file_path = r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\bgm\audio.mp3'
    audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
    print(sample_rate)
    print(audio_data.shape)
    feature_frames_per_beat = 512 * 16
    bpm = 120
    target_sample_rate = round(feature_frames_per_beat * bpm / 60)
    resampled_audio_data = torchaudio.functional.resample(audio_data, sample_rate, target_sample_rate)
    print(resampled_audio_data.shape)
    torchaudio.save(
        r'C:\Users\asus\coding\python\osu_auto_mapper\resources\data\bgm\resampled.wav',
        resampled_audio_data,
        target_sample_rate,
    )
