import torch

from util.result import metrics_util


def cal_acc_func(pred, label, output):
    """
    input should be tensors
    """
    return len(torch.where(pred == label)[0]) / len(label)


def cal_cm_func(pred, label, output):
    return metrics_util.get_epoch_confusion(pred.numpy(), label.numpy())