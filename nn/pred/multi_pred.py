import torch
from torch import nn


class MultiPred(nn.Module):
    def __init__(self):
        super(MultiPred, self).__init__()

    def forward(self, data):
        """
        cond_data(list of Tensor/Tensor): batch_size(sample_num), sample_label_num, class_pred_probability
        """
        if isinstance(data, torch.Tensor):
            return torch.argmax(data, dim=2)
        pred = [torch.argmax(data[sample_idx], dim=1) for sample_idx in range(len(data))]
        return pred
