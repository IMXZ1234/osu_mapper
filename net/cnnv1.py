import torch
from torch import nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(2, 64, kernel_size=17, stride=4, padding=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=17, stride=4, padding=8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True)
        )
        # last feature is speed_stars, which serves as an 'offset'
        self.fcn = nn.Conv1d(512 + 1, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, data):
        # input('?')
        x, speed_stars, valid_interval = data
        for i in range(len(x)):
            # 1, C, frame
            x[i] = self.cnn_layers(x[i].unsqueeze(dim=0))
            speed_stars_feature = torch.ones([1, 1, x[i].shape[2]], device=x[i].device)
            x[i] = torch.cat([x[i], speed_stars_feature], dim=1)
            x[i] = self.fcn(x[i]).permute([0, 2, 1]).squeeze(dim=0)
            # frame, possibilities
            if True in torch.isnan(x[i]):
                print('nan in x[%d]' % i)
        # (list)N, frame, possibilities
        return x, speed_stars, valid_interval
