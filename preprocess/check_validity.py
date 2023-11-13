import shutil
import os
import pickle
import traceback

from tqdm import tqdm
import numpy as np


def check_data_validity():
    data_dir_hhd = r'/home/data1/xiezheng/osu_mapper/preprocessed_v6'
    # data_dir_hhd = r'/home/data1/xiezheng/osu_mapper/preprocessed_v5'
    # data_dir_ssd = r'/home/xiezheng/data/preprocessed_v5'
    # for subdir in ['info', 'mel', 'label', 'label_idx', 'meta']:
    #     os.makedirs(os.path.join(data_dir_ssd, subdir), exist_ok=True)
    vis_dir = r'/home/data1/xiezheng/osu_mapper/vis'

    all_beatmapids = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(os.path.join(data_dir_hhd, 'info'))
    ]

    all_bad = []
    min_bad_x = []
    max_bad_x = []
    min_bad_y = []
    max_bad_y = []
    for beatmapid in tqdm(all_beatmapids):
        try:
            # with open(
            #     os.path.join(data_dir_hhd, 'info', str(beatmapid) + '.pkl'),
            #     'rb'
            # ) as f:
            #     total_mel_frames, beatmapset_id, first_occupied_snap, last_occupied_snap = pickle.load(f)

            is_bad = False
            with open(
                os.path.join(data_dir_hhd, 'label', str(beatmapid) + '.pkl'), 'rb'
            ) as f:
                label = pickle.load(f)
                x_coord, y_coord = label[:, 1], label[:, 2]

                x_coord = ((x_coord * (691 + 180) - 180) - 192) / 384
                y_coord = ((y_coord * (407 + 82) - 82) - 256) / 512

                bad_x_l1 = np.where(x_coord >= 1)[0]
                bad_x_s_1 = np.where(x_coord <= -1)[0]
                bad_y_l1 = np.where(y_coord >= 1)[0]
                bad_y_s_1 = np.where(y_coord <= -1)[0]

                if len(bad_x_l1) > 0:
                    is_bad = True
                    max_bad_x.append(x_coord[bad_x_l1])
                if len(bad_x_s_1) > 0:
                    is_bad = True
                    min_bad_x.append(x_coord[bad_x_s_1])

                if len(bad_y_l1) > 0:
                    is_bad = True
                    max_bad_y.append(y_coord[bad_y_l1])
                if len(bad_y_s_1) > 0:
                    is_bad = True
                    min_bad_y.append(y_coord[bad_y_s_1])

                if is_bad:
                    all_bad.append(beatmapid)
        except Exception:
            traceback.print_exc()
            print('bad beatmap %d' % beatmapid)
    print(all_bad)
    print(len(all_bad))
    print(max_bad_x, min_bad_x, max_bad_y, min_bad_y)
    with open(os.path.join(vis_dir, 'check_validity.pkl'), 'wb') as f:
        pickle.dump((max_bad_x, min_bad_x, max_bad_y, min_bad_y, all_bad), f)


if __name__ == '__main__':
    check_data_validity()
