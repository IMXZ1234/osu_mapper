from torch import nn
import torch
from torch.nn import functional as F


class MultiMSELoss(nn.Module):
    def __init__(self, weights=None):
        super(MultiMSELoss, self).__init__()
        self.weights = weights

    def forward(self, data, label):
        """
        cond_data(list of Tensor/Tensor): batch_size(sample_num), sample_label_num
        label(list of Tensor/Tensor): batch_size(sample_num), sample_label_num
        """
        if self.weights is None:
            self.weights = [1] * data[0].shape[-1]
        if isinstance(data, torch.Tensor):
            # print('loss cond_data shape')
            # print(cond_data.shape)
            # print('label cond_data shape')
            # print(label.shape)
            data = data.reshape(-1)
            label = label.reshape(-1)
            return F.mse_loss(data, label)
        # else pred and label are lists
        total_loss = []
        # print('len(cond_data)')
        # print(len(cond_data))
        for sample_idx in range(len(data)):
            sample_out = data[sample_idx]
            sample_label = label[sample_idx]
            # if True in torch.isnan(sample_out):
            #     print('nan in sample_out')
            total_loss.append(F.mse_loss(sample_out, sample_label))
        return torch.mean(torch.stack(total_loss))
