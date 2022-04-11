import torch
from torch import nn


class MultiPred(nn.Module):
    def __init__(self):
        super(MultiPred, self).__init__()

    def forward(self, data):
        """
        data(list of Tensor/Tensor): batch_size(sample_num), sample_label_num, class_pred_probability
        """
        if isinstance(data, torch.Tensor):
            return torch.argmax(data, dim=2)
        pred = [torch.argmax(data[sample_idx], dim=1) for sample_idx in range(len(data))]
        return pred


class ValidIntervalMultiPred(nn.Module):
    def __init__(self):
        super(ValidIntervalMultiPred, self).__init__()

    # def forward(self, gen):
    #     data, speed_stars, valid_interval = gen
    #     data = [torch.softmax(sample_data, dim=1) for sample_data in data]
    #     pred = [[] for _ in range(len(data))]
    #     for sample_idx in range(len(data)):
    #         sample_speed_stars = speed_stars[sample_idx]
    #         thresh = min(10, max(0, 10 - sample_speed_stars))
    #         sample_out = data[sample_idx]
    #         for snap_idx, snap_out in enumerate(sample_out):
    #             # snap_out[1] indicates possibility of having a hit_object at this snap
    #             # higher the speed_stars, smaller the thresh, the more likely a hit_object exists at that snap
    #             if snap_out[1] > thresh:
    #                 pred[sample_idx].append(1)
    #             else:
    #                 pred[sample_idx].append(0)
    #     return pred

    def forward(self, output):
        data, speed_stars, valid_interval = output
        data = [torch.softmax(sample_data, dim=1) for sample_data in data]
        pred = [torch.zeros(data[sample_idx].shape[0], device=data[0].device) for sample_idx in range(len(data))]
        for sample_idx in range(len(data)):
            sample_speed_stars, sample_valid_interval = speed_stars[sample_idx], valid_interval[sample_idx]
            pred[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]] = torch.argmax(data[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]], dim=1)
            # for snap_idx in range(sample_valid_interval[0], sample_valid_interval[1]):
            #     # snap_out[1] indicates possibility of having a hit_object at this snap
            #     sample_pred[snap_idx] = torch.argmax(sample_out[snap_idx])
        return pred
