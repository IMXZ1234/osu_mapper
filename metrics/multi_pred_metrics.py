import numpy as np
import torch

from util.result import metrics_util


def multi_pred_cal_acc_func(pred, label, output):
    """
    Calculates accuracy when a sample contains multiple predictions.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.reshape(-1)
        label = torch.cat(label).reshape(-1)
        return len(torch.where(pred == label)[0]) / len(pred)
    # else pred and label are lists
    return sum(len(torch.where(pred[i] == label[i])[0]) for i in range(len(pred))) / sum(len(sample_pred) for sample_pred in pred)


def multi_pred_cal_cm_func(pred, label, output):
    """
    Calculates confusion matrix when a sample contains multiple predictions.
    """
    if isinstance(pred, torch.Tensor):
        label = torch.cat(label)
        return metrics_util.get_epoch_confusion(pred.reshape(-1).numpy().astype(int),
                                                label.reshape(-1).numpy().astype(int))
    # else pred and label are lists
    pred = [pred[i].numpy() for i in range(len(pred))]
    label = [label[i].numpy() for i in range(len(label))]
    return metrics_util.get_epoch_confusion(np.concatenate(pred).astype(int),
                                            np.concatenate(label).astype(int))


def cal_acc_func_valid_itv(pred, label, output):
    valid_intervals = output[2]
    sample_total_snaps = [(valid_intervals[i][1] - valid_intervals[i][0]) for i in range(len(pred))]
    # print(pred)
    # print(label)
    sample_correct_snaps = [len(torch.where(pred[i][valid_intervals[i][0]:valid_intervals[i][1]] ==
                                            label[i][valid_intervals[i][0]:valid_intervals[i][1]])[0])
                            for i in range(len(pred))]
    # sample_correct_snaps = [0 for _ in range(len(pred))]
    # for sample_idx, sample_label in enumerate(label):
    #     sample_pred = pred[sample_idx]
    #     sample_valid_interval = valid_intervals[sample_idx]
    #     for snap_idx in range(sample_valid_interval[0], sample_valid_interval[1]):
    #         if sample_pred[snap_idx] == sample_label[snap_idx]:
    #             sample_correct_snaps[sample_idx] += 1
    # return [sample_correct_snaps[i] / sample_total_snaps[i] for i in range(len(pred))]
    return sum(sample_correct_snaps) / sum(sample_total_snaps)


def cal_cm_func_valid_itv(pred, label, output):
    valid_intervals = output[2]
    valid_pred = [pred[i][valid_intervals[i][0]:valid_intervals[i][1]].numpy() for i in range(len(pred))]
    valid_label = [label[i][valid_intervals[i][0]:valid_intervals[i][1]].numpy() for i in range(len(label))]
    return metrics_util.get_epoch_confusion(np.concatenate(valid_pred).astype(int),
                                            np.concatenate(valid_label).astype(int))