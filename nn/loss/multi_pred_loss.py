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
            data = data.reshape([-1, data.shape[-1]])
            label = label.reshape(-1)
            return F.cross_entropy(data, label,
                                   weight=torch.tensor(self.weight,
                                                       device=data.device,
                                                       dtype=torch.float))
        # else pred and label are lists
        total_loss = []
        for sample_idx in range(len(data)):
            sample_out = data[sample_idx]
            sample_label = label[sample_idx]
            total_loss.append(F.cross_entropy(sample_out, sample_label,
                                              weight=torch.tensor(self.weight,
                                                                  device=sample_out.device,
                                                                  dtype=torch.float)))
        return torch.mean(torch.stack(total_loss))


class ValidIntervalMultiPredLoss(nn.Module):
    def __init__(self, weight=(1, 1)):
        super(ValidIntervalMultiPredLoss, self).__init__()
        self.weight = weight

    def forward(self, output, label):
        data, speed_stars, valid_interval = output
        total_loss = []
        for sample_idx in range(len(data)):
            sample_speed_stars, sample_valid_interval = speed_stars[sample_idx], valid_interval[sample_idx]
            # thresh = min(10, max(0, 10 - sample_speed_stars))
            valid_sample_out = data[sample_idx][sample_valid_interval[0]:sample_valid_interval[1], :]
            valid_sample_label = label[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]]
            total_loss.append(F.cross_entropy(valid_sample_out, valid_sample_label,
                                              weight=torch.tensor(self.weight, device=valid_sample_out.device,
                                                                  dtype=torch.float)))
        return torch.mean(torch.stack(total_loss))


class MultiPredNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MultiPredNLLLoss, self).__init__()
        self.weight = weight

    def forward(self, data, label):
        """
        data(list of Tensor/Tensor): batch_size(sample_num), sample_label_num, class_pred_probability
        label(list of Tensor/Tensor): batch_size(sample_num), sample_label_num
        """
        if self.weight is None:
            self.weight = [1] * data[0].shape[-1]
        if isinstance(data, torch.Tensor):
            data = data.reshape([-1, data.shape[-1]])
            label = label.reshape(-1)
            return F.cross_entropy(data, label,
                                   weight=torch.tensor(self.weight,
                                                       device=data.device,
                                                       dtype=torch.float))
        # else pred and label are lists
        total_loss = []
        for sample_idx in range(len(data)):
            sample_out = data[sample_idx]
            sample_label = label[sample_idx]
            total_loss.append(F.nll_loss(sample_out, sample_label,
                                         weight=torch.tensor(self.weight,
                                                             device=sample_out.device,
                                                             dtype=torch.float)))
        return torch.mean(torch.stack(total_loss))
