import os
from datetime import timedelta

import numpy as np
import slider
import torchaudio
from matplotlib import pyplot as plt

from util.data import prepare_dataset, data_inspect, audio_osu_data


def divisor_distribution(divisor_list):
    possible_divisors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16]
    divisor_count = [0 for _ in range(len(possible_divisors))]
    back = [None for _ in range(max(possible_divisors) + 1)]
    for i, divisor in enumerate(possible_divisors):
        back[divisor] = i
    for divisor in divisor_list:
        divisor_count[back[divisor]] += 1
    return divisor_count


def count(list_in, min_value=None, max_value=None, bin_num=None, bin_len=None, value_on_interval=True, display=True):
    if min_value is None:
        min_value = np.min(list_in)
    if max_value is None:
        max_value = np.max(list_in)
    if value_on_interval:
        if bin_num is None:
            assert bin_len is not None
            bin_num = int((max_value - min_value) / bin_len)
        counts, bins, _ = plt.hist(list_in,
                                   bins=np.linspace(min_value, max_value, bin_num + 1, True))
    else:
        if bin_len is None:
            assert bin_num is not None
            bin_len = (max_value - min_value) / (bin_num - 1)
        if bin_num is None:
            assert bin_len is not None
            bin_num = int((max_value - min_value) / bin_len + 1)
        counts, bins, _ = plt.hist(list_in,
                                   bins=np.linspace(min_value - bin_len / 2,
                                                    max_value + bin_len / 2,
                                                    bin_num + 1, True))
    if display:
        plt.show()
        plt.clf()
    return counts, bins


def single_non_inherited_timepoints(beatmap):
    non_inherited_timepoint_num = 0
    for ti, timing_point in enumerate(beatmap.timing_points):
        if timing_point.bpm is not None:
            non_inherited_timepoint_num += 1
    return non_inherited_timepoint_num == 1


def cnnv1_criteria(beatmap):
    if not single_non_inherited_timepoints(beatmap):
        return False
    if beatmap.bpm_min() != beatmap.bpm_max():
        return False
    if beatmap.beat_divisor not in [1, 2, 4, 8]:
        return False
    return True


if __name__ == '__main__':
    # src_dir = os.getcwd()
    # proj_dir = os.path.dirname(src_dir)
    # print(proj_dir)
    # print(torchaudio.backend.list_audio_backends())
    # song_dir = prepare_dataset.OSUSongsDir()
    # song_dir.gen_index_file()
    # data = audio_osu_data.AudioOsuData.from_path(r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\local.pkl')
    data = audio_osu_data.AudioOsuData.from_path(r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\div2n_sninhtp.pkl')

    num = 0
    num1 = 0
    num2 = 0
    for beatmap in data.beatmaps:
        inherited_before_non_inherited = False
        first_timing_point_non_inherited = None
        first_timing_point = None
        f = False
        fn = False
        for time_point in beatmap.timing_points:
            if time_point.bpm is not None:
                first_timing_point_non_inherited = time_point.offset / timedelta(microseconds=1)
                if first_timing_point is None:
                    first_timing_point = first_timing_point_non_inherited
                f = True
                fn = True
            else:
                inherited_before_non_inherited = True
                first_timing_point = time_point.offset / timedelta(microseconds=1)
                f = True
            if f and fn:
                break
        if inherited_before_non_inherited:
            print('before')
            print(beatmap.beatmap_set_id)
            print(first_timing_point)
            print(first_timing_point_non_inherited)
        first_obj_time = beatmap.hit_objects()[0].time / timedelta(microseconds=1)
        # if first_obj_time < first_timing_point_non_inherited:
        #     print('obj')
        #     print(beatmap.beatmap_set_id)
        #     print(first_timing_point)
        #     print(first_timing_point_non_inherited)
        #     print(first_obj_time)
        #     num += 1
        start = min(first_timing_point, first_obj_time)
        if start < 0:
            print('start')
            if first_timing_point < 0:
                print('timing')
                num += 1
            if first_obj_time < 0:
                print('obj')
                num1 += 1
            if max(first_timing_point, first_obj_time) < 0:
                print('both')
                num2 += 1
            print(beatmap.beatmap_set_id)
            print(first_timing_point)
            print(first_timing_point_non_inherited)
            print(first_obj_time)
    print(num)
    print(num1)
    print(num2)
    # data_filter = prepare_dataset.DataFilter(data)
    # data = data_filter.keep_satisfied(cnnv1_criteria,
    #                            r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\div2n_sninhtp.pkl')
    # # data = audio_osu_data.AudioOsuData.from_path(r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\div2n_sninh_tp.pkl')
    # fd = prepare_dataset.FoldDivider(data)
    # fd.div_folds(r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\fold\div2n_sninhtp')
    # data_filter = prepare_dataset.DataFilter(data)
    # data_filter.keep_satisfied(has_multiple_non_inherited_timepoints,
    #                            r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\div2n_sninh_tp.pkl')
    # print(len(data.beatmaps))
    # beat_divisor_list = [beatmap.beat_divisor for beatmap in data.beatmaps]
    # print(beat_divisor_list)
    # divisor_count = divisor_distribution(beat_divisor_list)
    # timing_points_list = [beatmap.timing_points for beatmap in data.beatmaps]
    # non_inherited_timing_points_num = [0 for _ in range(len(timing_points_list))]
    # timing_points_num = []
    # for i, timing_points in enumerate(timing_points_list):
    #     timing_points_num.append(len(timing_points))
    #     for ti, timing_point in enumerate(timing_points):
    #         if timing_point.bpm is not None:
    #             non_inherited_timing_points_num[i] += 1
    # print(non_inherited_timing_points_num)
    # print(len(non_inherited_timing_points_num))
    # print(timing_points_num)
    # print(len(timing_points_num))
    # counts, bins = count(non_inherited_timing_points_num, bin_len=1, value_on_interval=False)
    # print(counts)
    # print(bins)
    # counts, bins = count(timing_points_num, bin_len=1, value_on_interval=False)
    # print(counts)
    # print(bins)
    # print(timing_points_list)
    # print(divisor_count)
    # plt.hist(beat_divisor_list)
    # plt.plot()
    # satisfied_index = data.get_satisfied(lambda beatmap: beatmap.beat_divisor in [1, 2, 4, 8])
    # satisfied_index = data.get_satisfied(lambda beatmap: beatmap.timing in [1, 2, 4, 8])
    # satisfied_index = data.get_satisfied(lambda beatmap: len(beatmap.timing_points))
    # print(satisfied_index)
    # print(len(satisfied_index))
    # satisfied_beatmaps = data.beatmaps_by_index(satisfied_index)
    # satisfied_audio_osu_list = data.audio_osu_list_by_index(satisfied_index)

    # data.save_index_file(satisfied_audio_osu_list, r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\commondiv.pkl')
    # data.save_beatmap_obj(satisfied_beatmaps, r'C:\Users\asus\coding\python\osu_beatmap_generator\resources\data\raw\commondiv_beatmaps.pkl')

    # inspector = data_inspect.DataInspector(data)
    # bpm_min_list = inspector.attr_distribution('bpm_min')
    # bpm_max_list = inspector.attr_distribution('bpm_max')
    # for i in range(len(bpm_min_list)):
    #     if bpm_max_list[i] != bpm_min_list[i]:
    #         print(data.beatmaps[i].beatmap_id)
    #         print(bpm_max_list[i])
    #         print(bpm_min_list[i])
    # prepare_raw.OSUSongsDir().gen_index_file()
    # prepare_raw.fetch_audio_osu()
    # lib = slider.library.Library(os.path.join(proj_dir, 'data/slider_db'))
    # prepare_raw.download_beatmap_with_specifications(lib, datetime.datetime(2021, 1, 1))
