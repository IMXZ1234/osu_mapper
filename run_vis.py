import os
from datetime import timedelta
from typing import Generator

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
    # cond_data, label
    items = [[], []]
    data, label = items
    # print('ids')
    # print(ids)
    cache = {0: None}
    sample_filter = filter.OsuTrainDataFilterGroup(
        [
            filter.ModeFilter(),
            filter.SnapDivisorFilter(),
            filter.SnapInNicheFilter(),
            filter.SingleUninheritedTimingPointFilter(),
            filter.SingleBMPFilter(),
            filter.BeatmapsetSingleAudioFilter(),
            filter.HitObjectFilter(),
        ]
    )
    for id_ in ids:
        record = train_db.get_record(id_, table_name)
        first_ho_snap = record[db.OsuDB.EXTRA_START_POS + 1]
        beatmap = pickle.loads(record[db.OsuDB.BEATMAP_POS])
        audio_path = os.path.join(r'./resources/data/mel', record[db.OsuDB.AUDIOFILENAME_POS])
        if not sample_filter.filter(beatmap, audio_path):
            continue
        yield beatmap, audio_path


def view_mel():
    for beatmap, audio_path in from_db_with_filter():
        with open(audio_path, 'rb') as f:
            audio_mel = pickle.load(f)
        print(audio_mel.shape)
        plt.imshow(audio_mel[0][:, :1000])
        print(audio_mel)
        plt.show()
        break


if __name__ == '__main__':
    view_dataset()
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
