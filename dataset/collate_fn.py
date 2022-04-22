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


def data_array_to_tensor(sample_list):
    """
    Any sample_data in should be [data(type=list), label, index],
    ndarray/tensor in data will be stacked as tensor
    other items in data will be simply stacked as a list.

    :param sample_list:
    :return:
    """
    # print('collate')
    data_item_num = len(sample_list[0][0])
    batch = [[[] for _ in range(data_item_num)], [], []]
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, label, index = sample_data
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray):
                item = torch.tensor(item, dtype=torch.float)
            batch[0][i].append(item)
        batch[1].append(label)
        batch[2].append(index)
    for i, item in enumerate(data):
        if isinstance(batch[0][i][0], torch.Tensor):
            batch[0][i] = torch.stack(batch[0][i])
    # print('batch[1]')
    # print(batch[1])
    # print([len(sample_label) for sample_label in batch[1]])
    batch[1] = torch.tensor(batch[1], dtype=torch.long)
    # print('data_array_to_tensor: data' + str(batch[0][0].shape))
    # print('data_array_to_tensor: label' + str(batch[1].shape))
    return batch


def data_array_to_tensor_inference(sample_list):
    """
    Any sample_data in should be [data(type=list), index],
    ndarray/tensor in data will be stacked as tensor
    other items in data will be simply stacked as a list.

    :param sample_list:
    :return:
    """
    # print('collate')
    data_item_num = len(sample_list[0][0])
    batch = [[[] for _ in range(data_item_num)], []]
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, index = sample_data
        for i, item in enumerate(data):
            if isinstance(item, np.ndarray):
                item = torch.tensor(item, dtype=torch.float)
            batch[0][i].append(item)
        batch[1].append(index)
    for i, item in enumerate(data):
        if isinstance(batch[0][i][0], torch.Tensor):
            batch[0][i] = torch.stack(batch[0][i])
    return batch


def output_collate_fn(epoch_output_list):
    """
    Flatten batches.
    """
    if isinstance(epoch_output_list[0], torch.Tensor):
        # if model output is a single tensor
        return torch.cat(epoch_output_list)
    # else model output is a list
    batch_out_item_num = len(epoch_output_list[0])
    collated = [[] for _ in range(batch_out_item_num)]
    for batch_idx, batch in enumerate(epoch_output_list):
        for item_idx in range(batch_out_item_num):
            if isinstance(batch[item_idx], (list, tuple)):
                collated[item_idx].extend(batch[item_idx])
            else:
                # a single item, most likely a tensor, with first dim as sample dim
                collated[item_idx].append(batch[item_idx])
    for i in range(batch_out_item_num):
        if isinstance(collated[i][0], torch.Tensor):
            # concatenate tensors at sample dim(the first dim)
            collated[i] = torch.cat(collated[i])
    return collated