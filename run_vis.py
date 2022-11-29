from datetime import timedelta

from vis import vis_model

import torch
import pickle
import numpy as np
import slider
from nn.dataset import dataset_util
from util import beatmap_util


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


def view_dataset():
    with open(
        r'C:\Users\asus\coding\python\osu_mapper\resources\data\fit\label_pos\label.pkl',
        'rb'
    ) as f:
        label_list = pickle.load(f)
    print(len(label_list))

    for label in label_list:
        pos_less_0 = np.where(label < 0)
        print(label[pos_less_0])
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


if __name__ == '__main__':
    beatmap = slider.Beatmap.from_path(
        r'C:\Users\asus\coding\python\osu_mapper\resources\solfa feat. Ceui - primal (Shikibe Mayu) [Expert].osu'
    )
    snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
    for ho in beatmap.hit_objects():
        if isinstance(ho, slider.beatmap.Slider):
            length = int((ho.end_time - ho.time) / timedelta(milliseconds=1) / snap_ms)
            if length <= 4:
                print(ho)
                print(length)
            pos = beatmap_util.slider_snap_pos(ho, length)
            pos_at_ticks = ho.tick_points
            for p in pos:
                if p.x < 0 or p.y < 0:
                    print(pos)
                    print(pos_at_ticks)
