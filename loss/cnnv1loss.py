from torch import nn
import torch
from torch.nn import functional as F


class CNNv1Loss(nn.Module):
    def __init__(self):
        super(CNNv1Loss, self).__init__()

    # def forward(self, output, label):
    #     data, speed_stars, valid_interval = output
    #     total_loss = 0
    #     data = [torch.softmax(sample_data, dim=1) for sample_data in data]
    #     for sample_idx in range(len(data)):
    #         sample_label = label[sample_idx]
    #         sample_speed_stars, sample_valid_interval = speed_stars[sample_idx], valid_interval[sample_idx]
    #         thresh = min(10, max(0, 10 - sample_speed_stars))
    #         sample_out = data[sample_idx]
    #         for snap_idx in range(sample_valid_interval[0], sample_valid_interval[1]):
    #             snap_out = sample_out[snap_idx]
    #             snap_label = sample_label[snap_idx]
    #             if snap_out[1] < thresh:
    #                 snap_label = 0
    #             total_loss += -torch.log(snap_out[1]) * snap_label
    #     return total_loss

    def forward(self, output, label):
        data, speed_stars, valid_interval = output
        total_loss = []
        # print('len(data)')
        # print(len(data))
        for sample_idx in range(len(data)):
            sample_speed_stars, sample_valid_interval = speed_stars[sample_idx], valid_interval[sample_idx]
            # thresh = min(10, max(0, 10 - sample_speed_stars))
            valid_sample_out = data[sample_idx][sample_valid_interval[0]:sample_valid_interval[1], :]
            valid_sample_label = label[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]]
            # if True in torch.isnan(valid_sample_out):
            #     print('nan in valid_sample_out')
            total_loss.append(F.cross_entropy(valid_sample_out, valid_sample_label,
                                              weight=torch.tensor([1, 1], device=valid_sample_out.device, dtype=torch.float)))
            # if True in torch.isnan(torch.stack(total_loss)):
            #     print('nan in total_loss')
            #     print(sample_idx)
            #     print('sample_valid_interval')
            #     print(sample_valid_interval)
            #     print('valid_sample_out')
            #     print(valid_sample_out)
            #     print(len(valid_sample_out))
            #     print('valid_sample_label')
            #     print(valid_sample_label)
            #     print(len(valid_sample_label))
        return torch.mean(torch.stack(total_loss))
