import torch
from torch import nn
from torch.nn import functional as F


class SegMultiPredMLP(nn.Module):

    def __init__(self, num_classes, extract_hidden_layer_num=2,
                 beat_feature_frames=16384, sample_beats=8, pad_beats=4,
                 snap_divisor=8):
        super().__init__()
        self.num_classes = num_classes
        self.snap_divisor = snap_divisor
        self.beat_feature_frames = beat_feature_frames - beat_feature_frames % self.snap_divisor
        self.sample_beats = sample_beats
        self.pad_beats = pad_beats
        self.sample_beats_padded = self.sample_beats + self.pad_beats * 2
        times = [2 ** i for i in range(extract_hidden_layer_num, -1, -1)]
        hidden = 512
        self.fc_layers = nn.ModuleList(
            # we have two channels for audio cond_data
            [nn.Linear(2 * self.sample_beats_padded * self.beat_feature_frames, times[0] * hidden)]
        )
        for i in range(extract_hidden_layer_num):
            self.fc_layers.append(
                nn.Linear(hidden * times[i], hidden * times[i + 1])
            )
        self.agg_fc = nn.Linear(hidden + 1, hidden)
        self.out_dim = self.snap_divisor * self.sample_beats
        self.out_fc = nn.Linear(hidden, self.num_classes * self.out_dim)

    def forward(self, x):
        # N, feature_num
        x, speed_stars = x
        x = x.reshape([x.shape[0], -1])
        # print('x.shape')
        # print(x.shape)
        # print('speed_stars.shape')
        # print(speed_stars)

        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            x = F.relu(x, inplace=True)
        speed_stars_feature = torch.tensor(speed_stars,
                                           device=x.device,
                                           dtype=torch.float)
        speed_stars_feature = speed_stars_feature.reshape([x.shape[0], -1])
        # print('x.shape')
        # print(x.shape)
        # print('speed_stars_feature.shape')
        # print(speed_stars_feature.shape)
        # cat along feature dim
        x = torch.cat([x, speed_stars_feature], dim=1)
        x = self.agg_fc(x)
        x = F.relu(x, inplace=True)
        x = self.out_fc(x)
        x = x.reshape([x.shape[0], self.out_dim, self.num_classes])
        # print('out x')
        # print(x.shape)

        return x
