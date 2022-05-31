import torch
from torch import nn


class SegMultiPredCNN(nn.Module):
    def __init__(self, num_classes, out_snaps=16*8):
        super().__init__()
        self.num_classes = num_classes
        self.cnn_layers = nn.Sequential(
            nn.BatchNorm1d(2),
            nn.Conv1d(2, 64, kernel_size=2049, stride=1, padding=1024),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(64),
            # nn.Conv1d(64, 64, kernel_size=1025, stride=2, padding=512),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=1025, stride=16, padding=512),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(128),
            # nn.Conv1d(128, 128, kernel_size=513, stride=4, padding=256),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=513, stride=16, padding=256),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(out_snaps),
        )
        # last feature is speed_stars, which serves as an 'offset'
        self.fcn = nn.Linear(256 + 1, 64)
        self.relu = nn.ReLU(inplace=True)
        self.fcn2 = nn.Linear(64, num_classes)

    def forward(self, data):
        # N, 2, 16384
        x, speed_stars = data
        # print('in x.shape')
        # print(x.shape)

        x = self.cnn_layers(x)
        # N, 256, 16*8
        batch_size, channels, out_snaps = x.shape
        x = x.permute([0, 2, 1]).reshape([batch_size * out_snaps, channels])
        # print('after conv.shape')
        # print(x.shape)
        # N * 16*8, 256
        speed_stars_feature = torch.tensor(speed_stars,
                                           device=x.device,
                                           dtype=torch.float)
        speed_stars_feature = speed_stars_feature.reshape([batch_size, 1, 1]).expand([batch_size, out_snaps, 1]).reshape([batch_size * out_snaps, 1])
        # print('speed_stars_feature.shape')
        # print(speed_stars_feature.shape)
        # cat along feature dim
        x = torch.cat([x, speed_stars_feature], dim=1)
        x = self.fcn(x)
        x = self.relu(x)
        x = self.fcn2(x)
        x = x.reshape([batch_size, out_snaps, self.num_classes])
        # print('out x')
        # print(x.shape)

        return x
