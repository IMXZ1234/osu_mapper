import torch
from torch import nn
from torch.nn import functional as F


class ValidIntervalCrossEntropy(nn.Module):
    def __init__(self, weight=(1, 1)):
        super(ValidIntervalCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, output, label):
        data, valid_intervals = output
        total_loss = []
        # print('len(cond_data)')
        # print(len(cond_data))
        for sample_idx in range(len(data)):
            sample_valid_interval = valid_intervals[sample_idx]
            # L, Class_num
            valid_sample_out = data[sample_idx][sample_valid_interval[0]:sample_valid_interval[1], :]
            valid_sample_label = label[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]]
            # if True in torch.isnan(valid_sample_out):
            #     print('nan in valid_sample_out')
            sample_loss = F.cross_entropy(valid_sample_out, valid_sample_label,
                                          weight=torch.tensor(self.weight, device=valid_sample_out.device,
                                                              dtype=torch.float))
            total_loss.append(sample_loss)
        return torch.mean(torch.stack(total_loss))


class ValidIntervalNLLLoss(nn.Module):
    def __init__(self, weight=(1, 1)):
        super(ValidIntervalNLLLoss, self).__init__()
        self.weight = weight

    def forward(self, output, label):
        data, valid_intervals = output
        total_loss = []
        # print('len(cond_data)')
        # print(len(cond_data))
        for sample_idx in range(len(data)):
            sample_valid_interval = valid_intervals[sample_idx]
            # L, Class_num
            valid_sample_out = data[sample_idx][sample_valid_interval[0]:sample_valid_interval[1], :]
            valid_sample_label = label[sample_idx][sample_valid_interval[0]:sample_valid_interval[1]]
            # if True in torch.isnan(valid_sample_out):
            #     print('nan in valid_sample_out')
            sample_loss = F.nll_loss(valid_sample_out, valid_sample_label,
                                     weight=torch.tensor(self.weight, device=valid_sample_out.device,
                                                         dtype=torch.float))
            # print('sample_loss %f' % sample_loss)
            # print('valid_sample_out')
            # print(valid_sample_out)
            # print('valid_sample_label')
            # print(valid_sample_label)
            total_loss.append(sample_loss)
        return torch.mean(torch.stack(total_loss))
