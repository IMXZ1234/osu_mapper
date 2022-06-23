import torch
import numpy as np

from util.result import metrics_util


def cal_acc_func_valid_itv(pred, label, output):
    valid_intervals = output[1]
    sample_total_snaps = [(valid_intervals[i][1] - valid_intervals[i][0]) for i in range(len(pred))]
    sample_correct_snaps = [len(torch.where(pred[i][valid_intervals[i][0]:valid_intervals[i][1]] ==
                                            label[i][valid_intervals[i][0]:valid_intervals[i][1]])[0])
                            for i in range(len(pred))]
    return sum(sample_correct_snaps) / sum(sample_total_snaps)


def cal_cm_func_valid_itv(pred, label, output):
    valid_intervals = output[1]
    valid_pred = [pred[i][valid_intervals[i][0]:valid_intervals[i][1]].numpy() for i in range(len(pred))]
    valid_label = [label[i][valid_intervals[i][0]:valid_intervals[i][1]].numpy() for i in range(len(label))]
    return metrics_util.get_epoch_confusion(np.concatenate(valid_pred).astype(int),
                                            np.concatenate(valid_label).astype(int))
