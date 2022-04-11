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
from dataset.dataset_util import hit_objects_to_label
import itertools
import bisect
import functools
import math


class SegMultiLabelDataset(Dataset):
    """
    Using index_file_path as reference.
    Load and convert mp3 file to wav on the fly to save disk space.
    Audio data are cut into segments(samples), for each segment there are
    multiple snaps whose hit object class are to be predicted.
    """

    def __init__(self, index_file_path, beatmap_obj_file_path=None,
                 beat_feature_frames=32768, snap_divisor=8, sample_beats=8, pad_beats=4,
                 multi_label=False,
                 shuffle=False):
        """
        pad_beats extra data is padded on both sides of one sample.
        Therefore, total beats for every sample is sample_beats + 2 * pad_beats,
        while total number of labels/(snaps whose hit object class are to be predicted)
        is sample_beats * snap_divisor.
        """
        super(SegMultiLabelDataset, self).__init__()
        self.audio_osu_data = audio_osu_data.AudioOsuData.from_path(index_file_path, beatmap_obj_file_path)
        if shuffle:
            self.audio_osu_data.shuffle()
        self.multi_label = multi_label
        # 8 is common snap divisor value
        self.snap_divisor = snap_divisor
        # adjust to multiples of snap_divisor
        self.beat_feature_frames = beat_feature_frames - beat_feature_frames % self.snap_divisor
        self.snap_feature_frames = self.beat_feature_frames // self.snap_divisor
        print('self.beat_feature_frames')
        print(self.beat_feature_frames)
        print('self.snap_feature_frames')
        print(self.snap_feature_frames)

        self.sample_beats = sample_beats
        self.pad_beats = pad_beats
        self.sample_beats_padded = self.sample_beats + self.pad_beats * 2

        # total beats for every sample
        self.sample_feature_frames = self.beat_feature_frames * self.sample_beats
        self.sample_feature_frames_padded = self.beat_feature_frames * self.sample_beats_padded

        self.sample_snaps = self.snap_divisor * self.sample_beats
        self.sample_snaps_padded = self.snap_divisor * self.sample_beats_padded
        # to calculate total number of samples
        self.beatmap_info_list = [(beatmap.bpm_min(),
                                   beatmap_util.get_first_hit_object_time_microseconds(beatmap),
                                   beatmap_util.get_last_hit_object_time_microseconds(beatmap))
                                  for beatmap in self.audio_osu_data.beatmaps]
        self.audio_snap_num = [functools.partial(SegMultiLabelDataset.cal_snap_num,
                                                 snap_divisor=self.snap_divisor,
                                                 )(*beatmap_info)
                                 for beatmap_info in self.beatmap_info_list]
        self.audio_sample_num = [snap_num // self.sample_snaps for snap_num in self.audio_snap_num]

        self.accumulate_audio_sample_num = list(itertools.accumulate(self.audio_sample_num))

        self.last_preprocessed_audio = None
        self.last_audio_label = None
        self.last_speed_stars = None
        self.last_audio_idx = -1
        self.last_audio_file_path = ''

    def __len__(self):
        return sum(self.audio_sample_num)

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
    def preprocess(audio_data, sample_rate,
                   bpm, start_time, end_time,
                   beat_feature_frames=32768,  # 512 * 64, 2**15
                   sample_beats=8,
                   pad_beats=4, ):
        """
        Resample the audio data to expand feature_frames_per_beat to target value
        Generally we have bpm of about 120 and sample rate of 44100Hz,
        which leads to about 22100 frames per beat.
        start_time, end_time in microsecond, should be aligned with snaps
        sample_rate in Hz
        Note we have two channels for audio data: audio_data.shape=[2, frame_num]
        """
        # print('bpm, start_time, end_time')
        # print(bpm, start_time, end_time)
        resample_rate = round(beat_feature_frames * bpm / 60)  # * feature_frames_per_second
        sample_feature_frames = sample_beats * beat_feature_frames
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
        # align end_time_frame with samples
        end_time_frame = (end_time_frame - start_time_frame) // sample_feature_frames * sample_feature_frames
        print('resampled_frame_num')
        print(resampled_frame_num)
        print('total beats num resample')
        print((end_time_frame - start_time_frame) // beat_feature_frames)

        # pad/trim at the start and end of the audio data
        pad_frames = pad_beats * beat_feature_frames
        pad_start_frame = pad_frames - start_time_frame
        pad_end_frame = pad_frames - (resampled_frame_num - 1 - end_time_frame)
        print('start_time_frame')
        print(start_time_frame)
        print('end_time_frame')
        print(end_time_frame)
        print('pad_start_frame')
        print(pad_start_frame)
        print('pad_end_frame')
        print(pad_end_frame)
        print('start_time_frame - end_time_frame')
        print(start_time_frame - end_time_frame)
        resampled_audio_data = resampled_audio_data[:, -min(pad_start_frame, 0):
                                                       -min(pad_end_frame, 0) + resampled_frame_num]
        # F.pad pads the last dim if param `pad` contains only two values
        resampled_audio_data = F.pad(resampled_audio_data,
                                     (max(pad_start_frame, 0), max(pad_end_frame, 0)),
                                     mode='replicate')
        print('resampled_audio_data.shape')
        print(resampled_audio_data.shape)
        return resampled_audio_data

    def __getitem__(self, index):
        """
        Samples are aligned with beats
        """
        audio_idx = bisect.bisect(self.accumulate_audio_sample_num, index)
        sample_idx = index - self.accumulate_audio_sample_num[audio_idx - 1] if audio_idx > 0 else index
        print('audio_idx')
        print(audio_idx)
        print('sample_idx')
        print(sample_idx)

        if self.last_audio_idx == audio_idx:
            # print('using cached')
            preprocessed_audio_data = self.last_preprocessed_audio
            audio_label = self.last_audio_label
            speed_stars = self.last_speed_stars
        else:
            # print('calculating')
            beatmap = self.audio_osu_data.beatmaps[audio_idx]
            audio_file_path = self.audio_osu_data.audio_osu_list[audio_idx][0]
            beatmap_info = self.beatmap_info_list[audio_idx]
            audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
            # print('beatmap_info')
            # print(beatmap_info)
            preprocessed_audio_data = SegMultiLabelDataset.preprocess(audio_data, sample_rate,
                                                                      *beatmap_info,
                                                                      self.beat_feature_frames,
                                                                      self.sample_beats,
                                                                      self.pad_beats)
            self.last_preprocessed_audio = preprocessed_audio_data

            snap_per_microsecond = self.beatmap_info_list[audio_idx][0] * self.snap_divisor / 60000000
            # print('snap_per_microsecond')
            # print(snap_per_microsecond)
            audio_label = hit_objects_to_label(beatmap,
                                               beatmap_info[1],  # first hit object time
                                               snap_per_microsecond,
                                               self.audio_snap_num[audio_idx],
                                               self.multi_label)
            self.last_audio_label = audio_label
            # print('len(audio_label)')
            # print(len(audio_label))

            speed_stars = beatmap_util.get_difficulty(beatmap)
            self.last_speed_stars = speed_stars

            self.last_audio_idx = audio_idx

        sample_feature_start_idx = sample_idx * self.sample_feature_frames
        sample_data = preprocessed_audio_data[:, sample_feature_start_idx:
                                                 sample_feature_start_idx + self.sample_feature_frames_padded]
        sample_snap_start_idx = sample_idx * self.sample_snaps
        sample_label = audio_label[sample_snap_start_idx:sample_snap_start_idx + self.sample_snaps]
        print('sample_data.shape')
        print(sample_data.shape)
        print('sample_label')
        print(len(sample_label))
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
