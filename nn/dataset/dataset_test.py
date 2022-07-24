import pickle
from datetime import timedelta

from preprocess import prepare_data_util
import slider
from util import beatmap_util, audio_util
from nn.dataset import dataset_util
import matplotlib.pyplot as plt


def speed_stars_dist():
    """
    View distribution of speed stars under osu songs dir
    """
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
    print(all_speed_stars)
    plt.hist(all_speed_stars)
    plt.show()
    with open(
            r'../../resources/test/all_speed_stars.pkl', 'wb'
    ) as f:
        pickle.dump(all_speed_stars, f)


def snap_divisor_dist():
    """
    View distribution of snaps per beat under osu songs dir
    """
    sd = prepare_data_util.OsuSongsDir()
    all_beat_divisors = []
    for beatmapset_dirname, beatmapset_dir_path, osu_filename, osu_file_path in sd.beatmaps():
        try:
            beatmap = slider.Beatmap.from_path(osu_file_path)
        except Exception:
            continue
        try:
            all_beat_divisors.append(beatmap.beat_divisor)
        except Exception:
            continue
    print(all_beat_divisors)
    plt.hist(all_beat_divisors)
    plt.show()
    with open(
            r'../../resources/test/all_beat_divisors.pkl', 'wb'
    ) as f:
        pickle.dump(all_beat_divisors, f)


def visualize_beatmap_hitobjects(audio_len, beatmap: slider.Beatmap):
    # song_end_time = hitobject_end_time(beatmap_util.get_last_hit_object_time_microseconds(
    #     beatmap, True, True, True, True
    # ))
    # first_ho_time = beatmap_util.get_last_hit_object_time_microseconds(
    #     beatmap, True, True, True, True
    # )
    plt.plot((0, audio_len), (0, 0), c='#ffffff', linewidth=0.1)
    plt.ylim((-1, 1))
    for ho in beatmap._hit_objects:
        time_start = ho.time / timedelta(milliseconds=1)
        if isinstance(ho, slider.beatmap.Circle):
            plt.scatter(time_start, 0, c='#ff0000', s=0.5)
    plt.show()


def visualize_osu_audio_hitobjects(audio_path, osu_path):
    """
    Wrapper over visualize_beatmap_hitobjects.
    """
    visualize_beatmap_hitobjects(
        audio_util.audio_len(audio_path),
        slider.Beatmap.from_path(osu_path)
    )


def hitobject_end_time(hitobject):
    if isinstance(hitobject, slider.beatmap.Circle):
        return hitobject.time
    else:
        return hitobject.end_time


def beatmap_hitobject_type_dist(beatmap):
    pass


def hitobject_type_dist():
    sd = prepare_data_util.OsuSongsDir()
    # no hitobject, circle, slider, spinner, holdnote
    hitobjects_num = [0] * 5
    for beatmapset_dirname, beatmapset_dir_path, osu_filename, osu_file_path in sd.beatmaps():
        print()


def is_snap_in_niche():
    sd = prepare_data_util.OsuSongsDir()
    not_in_niche = 0
    not_in_niche_ho = 0
    total_num = 0
    for beatmapset_dirname, beatmapset_dir_path, osu_filename, osu_file_path in sd.beatmaps():
        try:
            beatmap = slider.Beatmap.from_path(osu_file_path)
        except Exception:
            print('load error')
            continue
        if beatmap.bpm_min() != beatmap.bpm_max():
            print('bpm unsatisfied')
            continue
        if beatmap.beat_divisor > 8:
            print('beat divisor too large')
            continue
        total_num += 1
        bpm = beatmap.bpm_min()
        snap_milliseconds = 60000 / (bpm * 8)

        # tp_time = beatmap_util.get_first_uninherited_timing_point(beatmap).offset / timedelta(milliseconds=1)
        ho_time_list = [ho.time / timedelta(milliseconds=1) for ho in beatmap._hit_objects]
        snap_idx_list = [(ho_time - ho_time_list[0]) / snap_milliseconds for ho_time in ho_time_list]
        for snap_idx in snap_idx_list:
            if abs(snap_idx - round(snap_idx)) > 0.15:
                not_in_niche += 1
                print('not in niche')
                print(osu_file_path)
                print(snap_idx_list)
                break
    print('total_num')
    print(total_num)
    print('not_in_niche')
    print(not_in_niche)


def label_sample(snap_divisor=8):
    sd = prepare_data_util.OsuSongsDir()
    for beatmapset_dirname, beatmapset_dir_path, osu_filename, osu_file_path in sd.beatmaps():
        try:
            beatmap = slider.Beatmap.from_path(osu_file_path)
            if beatmap.beat_divisor != snap_divisor:
                continue
        except Exception:
            continue
        print(dataset_util.hitobjects_to_label_v2(beatmap, multi_label=True))
        break


if __name__ == '__main__':
    speed_stars_dist()
    # dataset_util.hitobjects_to_label_v2()
    # speed_stars_dist()
    # label_sample(8)
    # label_sample(4)
    # visualize_osu_audio_hitobjects(
    #     r'C:\Users\asus\AppData\Local\osu!\Songs\112197 Linked Horizon - Jiyuu no Tsubasa (TV Size)\OP2.mp3',
    #     r'C:\Users\asus\AppData\Local\osu!\Songs\112197 Linked Horizon - Jiyuu no Tsubasa (TV Size)\Linked Horizon - Jiyuu no Tsubasa (TV Size) (ritsu-tanaika) [Insane].osu',
    # )
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
