import numpy as np


def get_epoch_confusion(pred, true):
    assert len(pred) == len(true)
    max_label = np.max(true) + 1
    epoch_confusion_matrix = np.zeros([max_label, max_label], dtype=int)
    for i in range(len(pred)):
        epoch_confusion_matrix[true[i], pred[i]] += 1
    return epoch_confusion_matrix
