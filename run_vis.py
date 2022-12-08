import os
from datetime import timedelta
from typing import Generator

# import librosa

from vis import vis_model

import torch
from matplotlib import pyplot as plt
import pickle
import numpy as np
import slider
from nn.dataset import dataset_util
from util import beatmap_util
from preprocess import db, filter
np.set_printoptions(threshold=np.inf)


def power_to_db(specgram):
    return 10 * np.log10(specgram)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    # im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    im = axs.imshow(power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def view_osu():
    osu_path = r"C:\Users\asus\AppData\Local\osu!\Songs\1013342 Sasaki Sayaka - Sakura, Reincarnation\Sasaki Sayaka - Sakura, Reincarnation (Riana) [Mimari's Hard].osu"
    beatmap = slider.Beatmap.from_path(osu_path)
    print(beatmap.beat_divisor)
    snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
    label = dataset_util.hitobjects_to_label_v2(beatmap, snap_ms=snap_ms, multi_label=True)
    print(label)
    print([type(ho) for ho in beatmap._hit_objects])
    for ho in beatmap._hit_objects:
        if isinstance(ho, slider.beatmap.Slider):
            print(ho.num_beats)
            print(ho.time)
            print(ho.end_time)
            print(ho.position)


def view_model():
    model_path = r'D:\osu_mapper\resources\result\seqganv2\0.01\1\model_0_epoch_-1.pt'
    model = torch.load(model_path, map_location='cpu')
    gru_ih_l0 = model['gru.weight_ih_l0']
    print(gru_ih_l0.shape)
    # 128 + 128 + 514
    print(gru_ih_l0)
    embed, pos_embed, feature = gru_ih_l0[:, :128], gru_ih_l0[:, 128:256], gru_ih_l0[:, 256:]
    for item in [embed, pos_embed, feature]:
        print(torch.norm(item))
        print(torch.sum(torch.abs(item)) / torch.numel(item))


def find_single_elem(label, elem=2):
    total_len = 0
    pos = 0
    num_single_elem = 0
    while pos < len(label):
        while label[pos] == elem:
            total_len += 1
            pos += 1
            if pos >= len(label):
                break
        if total_len == 1:
            num_single_elem += 1
        total_len = 0
        pos += 1
    return num_single_elem


def view_dataset():
    with open(
        r'./resources/data/fit/label_pos/label.pkl',
        'rb'
    ) as f:
        label_list = pickle.load(f)
    print(len(label_list))

    for label in label_list:
        type_label = label[:, 0]
        num_single_elem = find_single_elem(type_label, 2)
        if num_single_elem > 0:
            print(num_single_elem)
            print(type_label)
    # with open(
    #     r'C:\Users\asus\coding\python\osu_mapper\resources\data\fit\label_pos\data.pkl',
    #     'rb'
    # ) as f:
    #     data_list = pickle.load(f)
    # print(len(data_list))
    # # print(data_list[0])
    # print(data_list[0].shape)
    # data = data_list[0]
    # print(np.mean(data))
    # print(np.max(data))
    # print(np.min(data))
    # for label in label_list:
    #     pos_less_0 = np.where(label < 0)
    #     print(label[pos_less_0])
    # total_over_1 = 0
    # max_x = 0
    # max_y = 0
    # for label in label_list:
    #     max_x = max(max_x, np.max(label[:, 1]))
    #     max_y = max(max_y, np.max(label[:, 2]))
    # print(max_x)
    # print(max_y)
    # min_x = 1
    # min_y = 1
    # for label in label_list:
    #     min_x = min(min_x, np.min(label[:, 1]))
    #     min_y = min(min_y, np.min(label[:, 2]))
    # print(min_x)
    # print(min_y)
        # if (label[:, 1:] == 1.).any():
        #     where_1 = np.where(label[:, 1:] == 1.)
        #     print(where_1)
        #     where_1 = (where_1[0], where_1[1] + 1)
        #     print(label[where_1])
            # print(label)

def view_beatmap():
    beatmap = slider.Beatmap.from_path(
        # r'C:\Users\asus\coding\python\osu_mapper\resources\solfa feat. Ceui - primal (Shikibe Mayu) [Expert].osu'
        # r'C:\Users\asus\AppData\Local\osu!\Songs\1774219 EPICA - Wings of Freedom\EPICA - Wings of Freedom (Luscent) [Alis Libertatis].osu'
        r'C:\Users\asus\AppData\Local\osu!\Songs\beatmap-638052960603827278-audio\zato - jiyuu (IMXZ123) [Insane].osu'
    )
    snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
    tp_list = beatmap.timing_points
    tp_list = [tp for tp in tp_list if tp.parent is not None]
    # next_tp_idx = 0
    # next_tp = tp_list[0]
    # current_sv = -100
    # print(next_tp.ms_per_beat)
    for ho in beatmap.hit_objects():
        if isinstance(ho, slider.beatmap.Slider):
            # if ho.time > next_tp.offset:
            #     next_tp_idx += 1
            #     current_sv = next_tp.ms_per_beat
            length = int((ho.end_time - ho.time) / timedelta(milliseconds=1) / snap_ms)
            # # if length <= 4:
            # #     print(ho)
            # #     print(length)
            pos = beatmap_util.slider_snap_pos(ho, length)
            pos_at_ticks = ho.tick_points
            for p in pos:
                if p.x < 0 or p.y < 0:
                    print(pos)
                    print(pos_at_ticks)


def from_db_with_filter():
    table_name = 'MAIN'
    train_db = db.OsuDB(
        r'./resources/data/osu_train_mel.db'
    )
    ids = train_db.all_ids(table_name)
    sample_filter = filter.OsuTrainDataFilterGroup(
        [
            filter.ModeFilter(),
            filter.SnapDivisorFilter(),
            filter.BeatmapsetSingleAudioFilter(),
            filter.HitObjectFilter(),
            filter.SingleUninheritedTimingPointFilter(),
            filter.SingleBMPFilter(),
            filter.SnapInNicheFilter(),
        ]
    )
    for id_ in ids:
        record = train_db.get_record(id_, table_name)
        # first_ho_snap = record[db.OsuDB.EXTRA_START_POS + 1]
        beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
        audio_path = os.path.join(r'./resources/data/mel', record[db.OsuDB.AUDIOFILENAME_POS])
        if not sample_filter.filter(beatmap, audio_path):
            continue
        yield beatmap, audio_path


def view_mel():
    i = 5
    for beatmap, audio_path in from_db_with_filter():
        with open(audio_path, 'rb') as f:
            audio_mel = pickle.load(f)
        print(audio_mel.shape)
        plot_spectrogram(audio_mel[0])
        # plt.imshow(audio_mel[0][:, :1000])
        # plt.show()
        # print(audio_mel)
        i -= 1
        if i < 0:
            break


def view_distribution(list_in, title=None, save=True):
    list_in = np.array(list_in)
    min_value, mean_value, max_value = np.min(list_in), np.mean(list_in), np.max(list_in)
    std = np.std(list_in)
    plt.hist(list_in)
    if title is None:
        title = 'plot'
    plt.title(title)
    if save:
        plt.savefig(r'./resources/vis/%s_%.4f_%.4f_%.4f_%.4f.png' % (title, min_value, mean_value, max_value, std))
    plt.show()
    with open('./resources/vis/meta/%s.pkl' % title, 'wb') as f:
        pickle.dump(list_in, f)


def ho_density_distribution():
    ho_density_list = []
    blank_proportion_list = []
    num_circle_list, num_slider_list = [], []
    circle_proportion = []
    overall_difficulty_list = []
    approach_rate_list = []
    for beatmap, audio_path in from_db_with_filter():
        assert isinstance(beatmap, slider.Beatmap)
        snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
        label = dataset_util.hitobjects_to_label_v2(beatmap, snap_ms=snap_ms)
        # total_snap_num = beatmap_util.get_total_snaps(beatmap)
        total_snap_num = len(label)
        ho_density_list.append(len(beatmap._hit_objects) / total_snap_num)
        num_circle, num_slider = 0, 0
        for ho in beatmap._hit_objects:
            if isinstance(ho, slider.beatmap.Circle):
                num_circle += 1
            elif isinstance(ho, slider.beatmap.Slider):
                num_slider += 1
        num_circle_list.append(num_circle)
        num_slider_list.append(num_slider)
        circle_proportion.append(float(num_circle) / float(num_circle + num_slider))
        blank_proportion_list.append(len(np.where(np.array(label) == 0)[0]) / len(label))
        overall_difficulty_list.append(beatmap.overall_difficulty)
        approach_rate_list.append(beatmap.approach_rate)
    view_distribution(ho_density_list, 'ho_density')
    view_distribution(blank_proportion_list, 'blank_proportion')
    view_distribution(num_circle_list, 'num_circle')
    view_distribution(num_slider_list, 'num_slider')
    view_distribution(circle_proportion, 'circle_proportion')
    view_distribution(overall_difficulty_list, 'overall_difficulty')
    view_distribution(approach_rate_list, 'approach_rate_list')


if __name__ == '__main__':
    ho_density_distribution()
        # assert isinstance(beatmap, slider.Beatmap)
        # try:
        #     all_speed_stars.append(beatmap.speed_stars())
        # except Exception:
        #     continue
    # plt.hist(all_speed_stars)
    # plt.show()
        # print(beatmap.speed_stars)
        # snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
        # label = dataset_util.hitobjects_to_label_with_pos(beatmap, snap_ms=snap_ms)
        # if label is not None:
        #     x, y = label[:, 1], label[:, 2]
        #     pos = np.where(x < 0)
        #     print(pos)
        #     # print(label[pos])
        #     print(label)
        #     input()
