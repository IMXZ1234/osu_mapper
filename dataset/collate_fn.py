import torch
import numpy as np


def non_array_to_list(sample_list):
    """
    Any sample_data in should be [data(type=list), label, index], any item in data will be simply stacked as a list.

    :param sample_list:
    :return:
    """
    data_item_num = len(sample_list[0][0])
    batch = [[[] for _ in range(data_item_num)], [], []]
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, label, index = sample_data
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray):
                item = torch.tensor(item, dtype=torch.float)
            batch[0][i].append(item)
        batch[1].append(torch.tensor(label, dtype=torch.long))
        batch[2].append(index)
    return batch
