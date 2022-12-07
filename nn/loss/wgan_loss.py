from torch import nn
import torch
from torch.nn import functional as F


class WGANLoss(nn.Module):
    def __init__(self, weight=None):
        super(WGANLoss, self).__init__()
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
