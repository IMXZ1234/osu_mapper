import torch
from torch.nn import functional as F
import numpy as np

from util.general_util import recursive_zip, recursive_to_tensor, recursive_stack


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
        array_label = np.array(label)
        if array_label.dtype == int:
            # classification label
            tensor_label = torch.tensor(array_label, dtype=torch.long)
        else:
            # regression label
            tensor_label = torch.tensor(array_label, dtype=torch.float)
        batch[1].append(tensor_label)
        batch[2].append(index)
    return batch


def collate_whole_seq(sample_list):
    """
    data in a batch may be of different length
    we simply put all data in a batch into a list
    padding/packing will be done later

    note label may be type int or float(if with hit_object position)
    data: seq_len, feature_dim
    ->batch (data, label, seq_len), added batch dim(dim=0)
    """
    batch = [[], [], []]
    max_len = max([sample_data[2] for sample_data in sample_list])
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, label, seq_len = sample_data
        data = torch.tensor(data, dtype=torch.float)
        # pad aat the end with 0
        data = F.pad(data, (0, max_len - seq_len, 0, 0), 'constant', 0)
        batch[0].append(data)
        label = np.array(label)
        if label.dtype == int:
            # classification label
            label = torch.tensor(label, dtype=torch.long)
        else:
            # regression label
            label = torch.tensor(label, dtype=torch.float)
        label = F.pad(label, (0, max_len - seq_len, 0, 0), 'constant', 0)
        batch[1].append(label)
        batch[2].append(seq_len)
    batch[0] = torch.stack(batch[0], dim=0)
    batch[1] = torch.stack(batch[1], dim=0)
    return batch


def default_collate(sample_list):
    batch = [[], [], []]
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, label, index = sample_data
        batch[0].append(torch.tensor(data, dtype=torch.float))
        array_label = np.array(label)
        if array_label.dtype == int:
            # classification label
            tensor_label = torch.tensor(array_label, dtype=torch.long)
        else:
            # regression label
            tensor_label = torch.tensor(array_label, dtype=torch.float)
        batch[1].append(tensor_label)
        batch[2].append(index)
    batch[0] = torch.stack(batch[0])
    batch[1] = torch.stack(batch[1])
    return batch


def default_collate_other(sample_list):
    """
    Any sample_data in should be [data(type=tensor), label, other], data will be stacked as tensor.
    'other' should be a dict, values in 'other' of one batch will be stacked into a list, with corresponding keys.

    :param sample_list:
    :return:
    """
    def dict_value_append(dict_in, dict_append):
        for k, v in dict_append.items():
            if k not in dict_in:
                dict_in[k] = [v]
            else:
                dict_in[k].append(v)

    batch = [[], [], dict()]
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, label, other = sample_data
        batch[0].append(torch.tensor(data, dtype=torch.float))
        array_label = np.array(label)
        if array_label.dtype == int:
            # classification label
            tensor_label = torch.tensor(array_label, dtype=torch.long)
        else:
            # regression label
            tensor_label = torch.tensor(array_label, dtype=torch.float)
        batch[1].append(tensor_label)
        dict_value_append(batch[2], other)
    # print(batch[2])
    batch[0] = torch.stack(batch[0])
    batch[1] = torch.stack(batch[1])
    return batch


def collate_same_data_len_seq2seq(sample_list):
    """
    Any sample_data in should be [data(type=tensor), label, index], data will be stacked as tensor.
    pad data and label at the end with 0 to same length(maximum length in the batch)
    has nested label in data

    :param sample_list:
    :return:
    """
    batch = [[], [], []]
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, label, index = sample_data
        batch[0].append(torch.tensor(data, dtype=torch.float))
        array_label = np.array(label)
        if array_label.dtype == int:
            # classification label
            tensor_label = torch.tensor(array_label, dtype=torch.long)
        else:
            # regression label
            tensor_label = torch.tensor(array_label, dtype=torch.float)
        batch[1].append(tensor_label)
        batch[2].append(index)
    # print(batch[2])
    # data
    data_len = [d.shape[0] for d in batch[0]]
    max_data_len = max(data_len)

    valid_interval = [(0, d.shape[0]) for d in batch[0]]
    batch[0] = [F.pad(d, (0, max_data_len - d.shape[0])) for d in batch[0]]

    label_len = [d.shape[0] for d in batch[0]]
    max_label_len = max(label_len)

    batch[1] = [F.pad(l, (0, max_label_len - l.shape[0])) for l in batch[1]]
    batch[1] = torch.stack(batch[1])
    
    batch[0] = [torch.stack(batch[0]), batch[1], valid_interval]

    # print('batch[0].shape')
    # print(len(batch[0][0]))
    # print(batch[0][1])
    # print(batch[0][2])
    # print('batch[1].shape')
    # print(batch[1].shape)
    return batch


def default_collate_other_rnn(sample_list):
    """
    Any sample_data in should be [data(type=tensor), label, other], data will be stacked as tensor.
    'other' should be a dict, values in 'other' of one batch will be stacked into a list, with corresponding keys.

    :param sample_list:
    :return:
    """
    def dict_value_append(dict_in, dict_append):
        for k, v in dict_append.items():
            if k not in dict_in:
                dict_in[k] = [v]
            else:
                dict_in[k].append(v)

    batch = [[], [], dict()]
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, label, other = sample_data
        # print('data')
        # print(data)
        # print('label')
        # print(label)
        # print('other')
        # print(other)
        batch[0].append(torch.tensor(data, dtype=torch.float))
        array_label = np.array(label)
        if array_label.dtype == int:
            # classification label
            tensor_label = torch.tensor(array_label, dtype=torch.long)
        else:
            # regression label
            tensor_label = torch.tensor(array_label, dtype=torch.float)
        batch[1].append(tensor_label)
        dict_value_append(batch[2], other)
    # print(batch[2])
    batch[0] = torch.stack(batch[0])
    batch[1] = torch.stack(batch[1])
    return batch


def default_collate_inference(sample_list):
    """
    Any sample_data in should be [data(type=list), label, index], any item in data will be simply stacked as a list.

    :param sample_list:
    :return:
    """
    batch = [[], []]
    for sample_data in sample_list:
        # sample_data = [data, label]
        data, index = sample_data
        batch[0].append(torch.tensor(data, dtype=torch.float))
        batch[1].append(index)
    batch[0] = torch.stack(batch[0])
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
    array_label = np.array(batch[1])
    if array_label.dtype == int:
        # classification label
        batch[1] = torch.tensor(array_label, dtype=torch.long)
    else:
        # regression label
        batch[1] = torch.tensor(array_label, dtype=torch.float)
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
    if epoch_output_list[0](first batch output) is a single tensor(if model output is a single tensor),
        cat into a single tensor [num_sample_in_all_batches, ...],
    elif model output is a list,
        cat every item in model output respectively like above and return them as a list.
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


def output_collate_fn_dfo(epoch_output_list):
    """
    Flatten batches.
    output = []
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


def tensor_list_stack(sample_list):
    """
    everything is tensor

    :param sample_list:
    :return:
    """
    collated = []
    # batch_data, batch_label, batch_index
    for i in range(len(sample_list[0])):
        collated.append(torch.stack([torch.tensor(sample_data[i], dtype=torch.float32) for sample_data in sample_list]))
    return collated


def tensor_list_stack_retain_type(sample_list):
    """
    everything is tensor

    :param sample_list:
    :return:
    """
    collated = []
    # batch_data, batch_label, batch_index
    for i in range(len(sample_list[0])):
        collated.append(torch.stack([torch.tensor(sample_data[i]) for sample_data in sample_list]))
    return collated


def tensor_list_recursive_stack(sample_list):
    """
    everything is tensor

    :param sample_list:
    :return:
    """
    collated = []
    for i in range(len(sample_list[0])):
        collated.append([sample_data[i] for sample_data in sample_list])
    for i in range(len(collated)):
        collated[i] = recursive_zip(collated[i])
        collated[i] = recursive_to_tensor(collated[i], dtype=torch.float32)
        collated[i] = recursive_stack(collated[i])
    return collated


def tensor_list_recursive_stack_retain_type(sample_list):
    """
    everything is tensor

    :param sample_list:
    :return:
    """
    collated = []
    for i in range(len(sample_list[0])):
        collated.append([sample_data[i] for sample_data in sample_list])
    for i in range(len(collated)):
        collated[i] = recursive_zip(collated[i])
        collated[i] = recursive_to_tensor(collated[i])
        collated[i] = recursive_stack(collated[i])
    return collated
