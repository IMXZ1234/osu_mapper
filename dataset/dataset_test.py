import pickle

from preprocess import prepare_data_util
import slider
from util import beatmap_util
import matplotlib.pyplot as plt


if __name__ == '__main__':
    sd = prepare_data_util.OsuSongsDir()
    all_speed_stars = []
    for beatmapset_dirname, beatmapset_dir_path, osu_filename, osu_file_path in sd.beatmaps():
        try:
            beatmap = slider.Beatmap.from_path(osu_file_path)
        except Exception:
            continue
        try:
            speed_stars = beatmap.speed_stars()
            all_speed_stars.append(speed_stars)
        except Exception:
            continue
    with open(
        r'C:\Users\asus\coding\python\osu_mapper\resources\test\all_speed_stars.pkl', 'wb'
    ) as f:
        pickle.dump(all_speed_stars, f)
    print(all_speed_stars)
    plt.hist(all_speed_stars)
    plt.show()
        # fho = beatmap_util.get_first_hit_object_time_microseconds(bp)
        # lho = beatmap_util.get_last_hit_object_time_microseconds(bp)
        # if bp.bpm_min() != bp.bpm_max():
        #     continue
        # uninherited_tp_num = 0
        # for tp in bp.timing_points:
        #     if tp.parent is None:
        #         uninherited_tp_num += 1
        # if uninherited_tp_num > 1:
        #     continue
        # bpm = bp.bpm_min()
        # tp = beatmap_util.get_first_uninherited_timing_point(bp)
        # if tp is None:
        #     continue
        # print('bpm %f, tp.bpm %f' % (bpm, tp.bpm))
        # sd = bp.beat_divisor
        # spm = bpm / sd
        # spms = spm / 60000000
        # print((lho - fho) * spms)
