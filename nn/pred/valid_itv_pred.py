import torch
from torch import nn


class ValidIntervalMultiPred(nn.Module):
    def __init__(self):
        super(ValidIntervalMultiPred, self).__init__()

    def forward(self, output):
        data, valid_intervals = output
        data = [torch.softmax(sample_data, dim=1) for sample_data in data]
        pred = [torch.zeros(data[sample_idx].shape[0], device=data[0].device) for sample_idx in range(len(data))]
        for sample_idx in range(len(data)):
            sample_valid_interval = valid_intervals[sample_idx]
            pred[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]] = torch.argmax(data[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]], dim=1)
        return pred
