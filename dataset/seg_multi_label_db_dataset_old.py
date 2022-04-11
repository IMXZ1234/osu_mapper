import bisect
import functools
import itertools
import os
import pickle
from collections import deque

import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.dataset_util import hit_objects_to_label
from util import audio_util, beatmap_util
from util.data import db


class SegMultiLabelDBDataset(Dataset):
    """
    Feed trainer with samples from OsuTrainDB
    """

    def __init__(self, db_path=db.OsuTrainDB.DEFAULT_DB_PATH, audio_dir=db.OsuTrainDB.DEFAULT_AUDIO_DIR,
                 table_name='TRAINFOLD1',
                 beat_feature_frames=32768, snap_divisor=8, sample_beats=8, pad_beats=4,
                 multi_label=False):
        """
        pad_beats extra data is padded on both sides of one sample.
        Therefore, total beats for every sample is sample_beats + 2 * pad_beats,
        while total number of labels/(snaps whose hit object class are to be predicted)
        is sample_beats * snap_divisor.
        """
        super(SegMultiLabelDBDataset, self).__init__()
        # self.dataset_name = table_name
        database = db.OsuTrainDB(db_path, connect=True)
        self.records = database.get_table_records(table_name)
        # self.beatmaps = [record[3] for record in self.records]
        self.beatmaps = [pickle.loads(record[3]) for record in self.records]
        self.audio_file_paths = [os.path.join(audio_dir, record[4]) for record in self.records]
        self.beatmaps_grouped_by_audio_file_path = {}
        # for beatmap, audio_file_path in zip(self.beatmaps, self.audio_file_paths):
        #     pass
        database.close()
        # print('self.audio_file_path')
        # print(self.audio_file_path)
        # if shuffle:
        #     self.audio_osu_data.shuffle()
        self.multi_label = multi_label
        # 8 is common snap divisor value
        self.snap_divisor = snap_divisor
        # adjust to multiples of snap_divisor
        self.beat_feature_frames = beat_feature_frames - beat_feature_frames % self.snap_divisor
        self.snap_feature_frames = self.beat_feature_frames // self.snap_divisor
        # print('self.beat_feature_frames')
        # print(self.beat_feature_frames)
        # print('self.snap_feature_frames')
        # print(self.snap_feature_frames)

        self.sample_beats = sample_beats
        self.pad_beats = pad_beats
        self.sample_beats_padded = self.sample_beats + self.pad_beats * 2

        # total beats for every sample
        self.sample_feature_frames = self.beat_feature_frames * self.sample_beats
        self.sample_feature_frames_padded = self.beat_feature_frames * self.sample_beats_padded

        self.sample_snaps = self.snap_divisor * self.sample_beats
        self.sample_snaps_pad = self.snap_divisor * self.pad_beats
        self.sample_snaps_padded = self.snap_divisor * self.sample_beats_padded
        # to calculate total number of samples
        self.beatmap_info_list = [(beatmap.bpm_min(),
                                   beatmap_util.get_first_hit_object_time_microseconds(beatmap),
                                   beatmap_util.get_last_hit_object_time_microseconds(beatmap))
                                  for beatmap in self.beatmaps]
        self.audio_snap_num = [functools.partial(SegMultiLabelDBDataset.cal_snap_num,
                                                 snap_divisor=self.snap_divisor,
                                                 )(*beatmap_info)
                               for beatmap_info in self.beatmap_info_list]
        self.audio_snap_per_microsecond = [beatmap_info[0] * self.snap_divisor / 60000000
                                           for beatmap_info in self.beatmap_info_list]
        self.audio_label = [hit_objects_to_label(beatmap,
                                                 beatmap_info[1],  # first hit object time
                                                 snap_per_microsecond,
                                                 audio_snap_num,
                                                 self.sample_snaps,
                                                 self.multi_label)
                            for beatmap,
                                beatmap_info,  # first hit object time
                                snap_per_microsecond,
                                audio_snap_num in
                            zip(self.beatmaps, self.beatmap_info_list, self.audio_snap_per_microsecond,
                                self.audio_snap_num)]
        self.speed_stars = [beatmap_util.get_difficulty(beatmap) for beatmap in self.beatmaps]
        # self.audio_sample_num = [snap_num // self.sample_snaps for snap_num in self.audio_snap_num]
        self.audio_sample_num = [snap_num // self.sample_snaps for snap_num in self.audio_snap_num]

        self.accumulate_audio_sample_num = list(itertools.accumulate(self.audio_sample_num))

        # lock_pool[self.dataset_name] = threading.Lock()
        # print(lock_pool)
        self.cache_len = 2
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
    def preprocess(resampled_audio_data,
                   bpm, start_time, end_time,
                   beat_feature_frames=32768,  # 512 * 64, 2**15
                   sample_beats=8,
                   audio_sample_num=None,
                   pad_beats=4):
        """
        Pad/trim resampled audio data in database for train use.
        start_time, end_time in microsecond, should be aligned with snaps
        sample_rate in Hz
        Note we have two channels for audio data: audio_data.shape=[2, frame_num]
        """
        # print('bpm, start_time, end_time')
        # print(bpm, start_time, end_time)
        resampled_frame_num = resampled_audio_data.shape[1]
        resample_rate = round(beat_feature_frames * bpm / 60)  # * feature_frames_per_second
        sample_feature_frames = sample_beats * beat_feature_frames
        resample_rate_MHz = resample_rate / 1000000
        start_time_frame = max(0, round(start_time * resample_rate_MHz))
        end_time_frame = min(resampled_frame_num, round(end_time * resample_rate_MHz))
        # there exists a hit object exactly at end time, length of preprocessed audio should
        # go beyond end_time_frame to include features for the last hit object
        audio_calculated_sample_num = (end_time_frame - start_time_frame) // sample_feature_frames + 1
        # apply sample number check
        if audio_sample_num is not None:
            if audio_calculated_sample_num < audio_sample_num:
                # print('audio sample num larger than calculated!')
                # print('audio_calculated_sample_num')
                # print(audio_calculated_sample_num)
                # print('audio_sample_num')
                # print(audio_sample_num)
                audio_calculated_sample_num = audio_sample_num
        # align end_time_frame with samples
        end_time_frame = audio_calculated_sample_num * sample_feature_frames + start_time_frame

        # pad/trim at the start and end of the audio data
        pad_frames = pad_beats * beat_feature_frames
        pad_start_frame = pad_frames - start_time_frame
        pad_end_frame = pad_frames - (resampled_frame_num - end_time_frame)
        # print('pad_frames')
        # print(pad_frames)
        # print('start_time_frame')
        # print(start_time_frame)
        # print('end_time_frame')
        # print(end_time_frame)
        # print('pad_start_frame')
        # print(pad_start_frame)
        # print('pad_end_frame')
        # print(pad_end_frame)
        # print('start_time_frame - end_time_frame')
        # print(start_time_frame - end_time_frame)
        resampled_audio_data = resampled_audio_data[:,
                               -min(pad_start_frame, 0):min(pad_end_frame, 0) + resampled_frame_num]
        # print('resampled_audio_data.shape after crop')
        # print(resampled_audio_data.shape)
        # F.pad pads the last dim if param `pad` contains only two values
        resampled_audio_data = F.pad(resampled_audio_data,
                                     (max(pad_start_frame, 0), max(pad_end_frame, 0)),
                                     mode='replicate')
        # print('resampled_audio_data.shape')
        # print(resampled_audio_data.shape)
        return resampled_audio_data

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

        cached = self.cache_operation(audio_file_path, op='get')
        if cached is not None:
            # print('using cached')
            # print(audio_file_path)
            preprocessed_audio_data, audio_label, speed_stars = cached
        else:
            # print('calculating')
            # print(audio_file_path)
            beatmap_info = self.beatmap_info_list[audio_idx]
            # this is resampled data
            audio_data, sample_rate = audio_util.audioread_get_audio_data(audio_file_path)
            # print('beatmap_info')
            # print(beatmap_info)
            preprocessed_audio_data = SegMultiLabelDBDataset.preprocess(audio_data,
                                                                        *beatmap_info,
                                                                        self.beat_feature_frames,
                                                                        self.sample_beats,
                                                                        self.audio_sample_num[audio_idx],
                                                                        self.pad_beats)

            audio_label = self.audio_label[audio_idx]
            speed_stars = self.speed_stars[audio_idx]
            self.cache_operation(audio_file_path, (preprocessed_audio_data, audio_label, speed_stars), op='put')
            # print('len(audio_label)')
            # print(len(audio_label))
            # print('preprocessed_audio_data.shape')
            # print(preprocessed_audio_data.shape)


        sample_feature_start_idx = sample_idx * self.sample_feature_frames
        sample_data = preprocessed_audio_data[:, sample_feature_start_idx:
                                                 sample_feature_start_idx + self.sample_feature_frames_padded]
        # print('data itv')
        # print('%d %d' % (sample_feature_start_idx, sample_feature_start_idx + self.sample_feature_frames_padded))
        sample_snap_start_idx = sample_idx * self.sample_snaps
        # print('label itv')
        # print('%d %d' % (sample_snap_start_idx, sample_snap_start_idx + self.sample_snaps))
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
