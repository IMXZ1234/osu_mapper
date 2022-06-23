import numpy as np
import torch

from util.result import metrics_util


def multi_pred_cal_acc_func(pred, label, output):
    """
    Calculates accuracy when a sample contains multiple predictions.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.reshape(-1)
        label = label.reshape(-1)
        return len(torch.where(pred == label)[0]) / len(pred)
    # else pred and label are lists
    return sum(len(torch.where(pred[i] == label[i])[0]) for i in range(len(pred))) / sum(len(sample_pred) for sample_pred in pred)


def multi_pred_cal_cm_func(pred, label, output):
    """
    Calculates confusion matrix when a sample contains multiple predictions.
    """
    if isinstance(pred, torch.Tensor):
        return metrics_util.get_epoch_confusion(pred.reshape(-1).numpy().astype(int),
                                                label.reshape(-1).numpy().astype(int))
    # else pred and label are lists
    pred = [pred[i].numpy() for i in range(len(pred))]
    label = [label[i].numpy() for i in range(len(label))]
    return metrics_util.get_epoch_confusion(np.concatenate(pred).astype(int),
                                            np.concatenate(label).astype(int))
