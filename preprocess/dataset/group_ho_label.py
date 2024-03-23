import os
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':
    root_dir = r'/home/xiezheng/data/preprocessed_v7'
    vis_dir = r'/home/xiezheng/osu_mapper/resources/vis'
    label_dir = os.path.join(root_dir, 'label')
    info_dir = os.path.join(root_dir, 'info')
    group_size = 4
    grouped_label_dir = os.path.join(root_dir, 'label_group_size%d' % group_size)
    os.makedirs(grouped_label_dir, exist_ok=True)
    idx2labelgroup_filepath = os.path.join(root_dir, 'idx2labelgroup_size%d.pkl' % group_size)
    labelgroup2idx_filepath = os.path.join(root_dir, 'labelgroup2idx_size%d.pkl' % group_size)
    counter_filepath = os.path.join(root_dir, 'labelgroup_counter_size%d.pkl' % group_size)
    existent_groups = {}
    label_filename_pair_list = []
    for filename in tqdm(os.listdir(label_dir), ncols=80):
        filepath = os.path.join(label_dir, filename)
        info_filepath = os.path.join(info_dir, filename)
        if not os.path.exists(info_filepath):
            continue
        with open(info_filepath, 'rb') as f:
            beat_divisor = pickle.load(f)[-1]
        # if beat_divisor != 4:
        #     continue

        with open(filepath, 'rb') as f:
            label = pickle.load(f)[:, 0].astype(int)
        label_filename_pair_list.append((label, filename))
        for i in range(0, len(label), group_size):
            label_segment = tuple(label[i: i + group_size].tolist())
            if label_segment not in existent_groups:
                existent_groups[label_segment] = 1
            else:
                existent_groups[label_segment] += 1

    # print(existent_groups)
    print(len(existent_groups))
    sorted_dist = list(sorted(existent_groups.items(), key=lambda x: x[1]))
    print(sorted_dist)
    idx2labelgroup, occurrence = list(zip(*sorted_dist))
    vis_dist_filepath = os.path.join(vis_dir, 'label_group_dist_size4.jpg')
    plt.figure()
    plt.bar(list(range(len(existent_groups))), occurrence)
    plt.savefig(vis_dist_filepath)
    vis_dist_filepath = os.path.join(vis_dir, 'label_group_dist_log_size4.jpg')
    plt.figure()
    plt.bar(list(range(len(existent_groups))), np.log10(occurrence))
    plt.savefig(vis_dist_filepath)
    with open(counter_filepath, 'wb') as f:
        pickle.dump(sorted_dist, f)

    labelgroup2idx = {
        idx2labelgroup[i]: i
        for i in range(len(idx2labelgroup))
    }

    with open(idx2labelgroup_filepath, 'wb') as f:
        pickle.dump(idx2labelgroup, f)

    with open(labelgroup2idx_filepath, 'wb') as f:
        pickle.dump(labelgroup2idx, f)

    for label, filename in tqdm(label_filename_pair_list, ncols=80):
        tgt_filepath = os.path.join(grouped_label_dir, filename)
        if len(label) % group_size != 0:
            print('label size not multiples of group size!', filename, len(label))
            continue
        reshaped_label = label.reshape([-1, group_size])
        grouped_label = np.array(
            [labelgroup2idx[tuple(reshaped_label[i].tolist())]
            for i in range(reshaped_label.shape[0])]
        )
        with open(tgt_filepath, 'wb') as f:
            pickle.dump(grouped_label, f)
