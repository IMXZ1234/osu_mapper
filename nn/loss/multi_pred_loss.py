from torch import nn
import torch
from torch.nn import functional as F


class MultiPredLoss(nn.Module):
    def __init__(self, weight=None):
        super(MultiPredLoss, self).__init__()
        self.weight = weight

    def forward(self, data, label):
        """
        data(list of Tensor/Tensor): batch_size(sample_num), sample_label_num, class_pred_probability
        label(list of Tensor/Tensor): batch_size(sample_num), sample_label_num
        """
        if self.weight is None:
            self.weight = [1] * data[0].shape[-1]
        if isinstance(data, torch.Tensor):
            # print('loss cond_data shape')
            # print(cond_data.shape)
            # print('label cond_data shape')
            # print(label.shape)
            data = data.reshape([-1, data.shape[-1]])
            label = label.reshape(-1)
            return F.cross_entropy(data, label,
                                   weight=torch.tensor(self.weight,
                                                       device=data.device,
                                                       dtype=torch.float))
        # else pred and label are lists
        total_loss = []
        # print('len(cond_data)')
        # print(len(cond_data))
        for sample_idx in range(len(data)):
            sample_out = data[sample_idx]
            sample_label = label[sample_idx]
            # if True in torch.isnan(sample_out):
            #     print('nan in sample_out')
            total_loss.append(F.cross_entropy(sample_out, sample_label,
                                              weight=torch.tensor(self.weight,
                                                                  device=sample_out.device,
                                                                  dtype=torch.float)))
        return torch.mean(torch.stack(total_loss))


class ValidIntervalMultiPredLoss(nn.Module):
    def __init__(self, weight=(1, 1)):
        super(ValidIntervalMultiPredLoss, self).__init__()
        self.weight = weight

    # def forward(self, gen, label):
    #     cond_data, speed_stars, valid_interval = gen
    #     total_loss = 0
    #     cond_data = [torch.softmax(sample_data, dim=1) for sample_data in cond_data]
    #     for sample_idx in range(len(cond_data)):
    #         sample_label = label[sample_idx]
    #         sample_speed_stars, sample_valid_interval = speed_stars[sample_idx], valid_interval[sample_idx]
    #         thresh = min(10, max(0, 10 - sample_speed_stars))
    #         sample_out = cond_data[sample_idx]
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
        # print('len(cond_data)')
        # print(len(cond_data))
        for sample_idx in range(len(data)):
            sample_speed_stars, sample_valid_interval = speed_stars[sample_idx], valid_interval[sample_idx]
            # thresh = min(10, max(0, 10 - sample_speed_stars))
            valid_sample_out = data[sample_idx][sample_valid_interval[0]:sample_valid_interval[1], :]
            valid_sample_label = label[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]]
            # if True in torch.isnan(valid_sample_out):
            #     print('nan in valid_sample_out')
            total_loss.append(F.cross_entropy(valid_sample_out, valid_sample_label,
                                              weight=torch.tensor(self.weight, device=valid_sample_out.device,
                                                                  dtype=torch.float)))
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
