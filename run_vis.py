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

    # for k, v in model.items():
    #     print(k)
    #     print(v.shape)


if __name__ == '__main__':
    beatmap = slider.Beatmap.from_path(
        r'C:\Users\asus\coding\python\osu_mapper\resources\solfa feat. Ceui - primal (Shikibe Mayu) [Expert].osu'
    )
    aligned_ms = beatmap_util.get_first_hit_object_time_milliseconds(
        beatmap,
        True,
        True,
        True,
        True,
    )
    ho = beatmap.hit_objects()[0]
    snap_ms = beatmap_util.get_snap_milliseconds(beatmap, 8)
    time, end_time = ho.time / timedelta(milliseconds=1), ho.end_time / timedelta(milliseconds=1)
    snap_idx = round((time - aligned_ms) / snap_ms)
    end_snap_idx = round((end_time - aligned_ms) / snap_ms)
    print(end_time - time)
    print(ho.length)
    print(ho.repeat)
    print(beatmap.slider_multiplier)
    print(snap_ms)
    print(snap_idx)
    print(end_snap_idx)
    print(ho)
    print(ho.curve)
    print(beatmap_util.slider_snap_pos(ho, end_snap_idx - snap_idx))
    # view_model()
    # with open(
    #         r'C:\Users\asus\coding\python\osu_mapper\resources\result\cganv3_pe_sample_beats_48\0.1\1\trainepoch_gen_output_list_epoch24.pkl',
    #         'rb'
    # ) as f:
    #     gen_out = pickle.load(f)
    # print(torch.argmax(gen_out[12], dim=1))
    # print(len(gen_out[0]))
            # break
#     with open(
# r'C:\Users\asus\coding\python\osu_mapper\resources\data\fit\rnnv3_nolabel\train1_label.pkl',
#             'rb') as f:
#         labels = pickle.load(f)
#     pattern_len = 4
#     for sample_label in labels:
#         print(sample_label.tolist())
#         break
#         sample_label = sample_label[:(len(sample_label)//pattern_len)*pattern_len].reshape([-1, pattern_len])
#         for k in sample_label:
#             if np.equal(np.array([1, 1, 1, 1]), sample_label[k]).all():
#                 # print(sample_label)
#                 print('at')
#                 print(k)
                # print('at %d' % k * 4)
    # print(torch.argmax(gen_out[12], dim=1))
    # print(len(gen_out[0]))
    # mv = vis_model.ModelParamViewer(
    #     r'resources/config/inference/rnnv1_lr0.01.yaml',
    # )
