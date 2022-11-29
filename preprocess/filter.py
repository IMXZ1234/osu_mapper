import os
from abc import abstractmethod
from datetime import timedelta

import slider

from util import beatmap_util


class OsuTrainDataFilter:
    """
    This class is intended to work together with OsuTrainDB.
    Filters out invalid train sample.
    """
    def __init__(self):
        pass

    @abstractmethod
    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        """
        If sample should be filtered out, return False.
        """
        raise NotImplementedError


class OsuTrainDataFilterGroup(OsuTrainDataFilter):
    def __init__(self, data_filters: list = None):
        super().__init__()
        self.data_filters = data_filters

    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        if self.data_filters is None:
            return True
        for data_filter in self.data_filters:
            if not data_filter.filter(beatmap, audio_file_path):
                return False
        return True


class HitObjectFilter(OsuTrainDataFilter):
    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        if len(beatmap.hit_objects()) == 0:
            return False
        return True


class BeatmapsetSingleAudioFilter(OsuTrainDataFilter):
    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        # if there are multiple audios for a single beatmap,
        # according to the naming principle of OsuTrainDB,
        # second, third... audio will be named as xx_1.mp3, xx_2.mp3...
        if '_' in os.path.basename(audio_file_path):
            return False
        return True


class SingleBMPFilter(OsuTrainDataFilter):
    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        if beatmap.bpm_min() != beatmap.bpm_max():
            # print('beatmap.bpm_min != beatmap.bpm_max')
            # print('beatmap.beatmap_set_id')
            # print(beatmap.beatmap_set_id)
            # print('beatmap.beatmap_id')
            # print(beatmap.beatmap_id)
            # print('beatmap.bpm_min')
            # print(beatmap.bpm_min())
            # print('beatmap.bpm_max')
            # print(beatmap.bpm_max())
            return False
        return True


class SingleUninheritedTimingPointFilter(OsuTrainDataFilter):
    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        uninherited_tp_num = 0
        for tp in beatmap.timing_points:
            if tp.parent is None:
                uninherited_tp_num += 1
        if uninherited_tp_num > 1:
            # print('uninherited_tp_num > 1')
            # print('beatmap.beatmap_set_id')
            # print(beatmap.beatmap_set_id)
            # print('beatmap.beatmap_id')
            # print(beatmap.beatmap_id)
            # print('uninherited_tp_num')
            # print(uninherited_tp_num)
            return False
        return True


class SnapInNicheFilter(OsuTrainDataFilter):
    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        bpm = beatmap.bpm_min()
        snap_milliseconds = 60000 / (bpm * 8)
        # tp_time = beatmap_util.get_first_uninherited_timing_point(beatmap).offset / timedelta(milliseconds=1)
        first_ho_time = beatmap._hit_objects[0].time / timedelta(milliseconds=1)
        for ho in beatmap._hit_objects:
            ho_time = ho.time / timedelta(milliseconds=1)
            snap_idx = (ho_time - first_ho_time) / snap_milliseconds
            if abs(snap_idx - round(snap_idx)) > 0.1:
                print('not in niche!')
                print(beatmap.beatmap_set_id)
                print()
                return False
        return True


class SnapDivisorFilter(OsuTrainDataFilter):
    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        if beatmap.beat_divisor > 8:
            print('beat_divisor > 8: %d' % beatmap.beat_divisor)
            print('beatmap.beatmap_set_id %d' % beatmap.beatmap_set_id)
            print()
            return False
        if beatmap.beat_divisor % 2 != 0:
            print('beat_divisor is odd: %d' % beatmap.beat_divisor)
            print('beatmap.beatmap_set_id %d' % beatmap.beatmap_set_id)
            print()
            return False
        return True


class ModeFilter(OsuTrainDataFilter):
    def filter(self, beatmap: slider.Beatmap, audio_file_path):
        return beatmap.mode == 0
