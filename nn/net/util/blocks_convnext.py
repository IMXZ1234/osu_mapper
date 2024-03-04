import torch

from nn.net.util.transformer_modules import PosEncoding
from torch import nn
from torch.nn import functional as F


class LayerNorm1D(nn.Module):
    def __init__(self, normalized_shape, **kwargs):
        super(LayerNorm1D, self).__init__()
        self.norm_layer = nn.LayerNorm(normalized_shape, **kwargs)

    def forward(self, x):
        """
        x: N, C, L
        """
        # swap channel dim to the last dim before layernorm
        x = torch.transpose(self.norm_layer(torch.transpose(x, 1, -1)), 1, -1)
        return x


def parse_norm(channels, norm='LN'):
    if norm is None:
        norm_layer = nn.Identity()
    elif norm == 'BN':
        norm_layer = nn.BatchNorm1d(channels)
    elif norm == 'LN':
        norm_layer = LayerNorm1D(channels)
    else:
        raise ValueError('unknown normalize')
    return norm_layer


class TransformerStem(nn.Module):
    def __init__(self, in_channels, seq_len, num_layers=3):
        super().__init__()
        self.pos_encoding = PosEncoding(seq_len, in_channels, batch_first=True)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                in_channels,
                8,
                in_channels * 2,
                dropout=0,
                activation=F.leaky_relu,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        """x: n, c, l"""
        # -> n, l, c
        x = x.permute(0, 2, 1)
        res = x
        x = self.pos_encoding(x)
        for layer in self.encoder:
            x = layer(x)
        x = x + res
        # -> n, c, l
        x = x.permute(0, 2, 1)
        return x


class CNNAttStem(nn.Module):
    def __init__(self, in_channels, seq_len, num_layers=3, norm='LN'):
        super().__init__()
        self.pos_encoding = PosEncoding(seq_len, in_channels, batch_first=True)
        self.att = nn.ModuleList([
            nn.MultiheadAttention(
                in_channels,
                8,
                dropout=0,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.cnn = nn.ModuleList([
            ConvNeXtBlock1D(
                in_channels,
                in_channels,
                seq_len,
                stride=1,
                norm=norm
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        """x: n, c, l"""
        for att_layer, cnn_layer in zip(self.att, self.cnn):
            # -> n, l, c
            encoded = self.pos_encoding(x.permute(0, 2, 1))
            # -> n, c, l
            x = att_layer(encoded, encoded, encoded)[0].permute(0, 2, 1) + cnn_layer(x)
        return x


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, bottleneck_channels=None,
                 stride=1, kernel_size=5,
                 norm='LN', dropout=0, residual=True):
        super().__init__()
        self.residual = residual
        if (not self.residual) or ((in_channels == out_channels) and (stride == 1)):
            self.short_cut = nn.Identity()
        else:
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
                parse_norm(out_channels, norm)
            )
        if bottleneck_channels is None:
            bottleneck_channels = 2 * in_channels
        self.norm_bottleneck = parse_norm(in_channels, norm)
        padding = kernel_size // 2
        self.conv_in = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv_bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, padding=0, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        """x: n, c, l"""
        res = self.short_cut(x)
        x = self.conv_in(x)
        x = self.norm_bottleneck(x)
        x = self.conv_bottleneck(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv_out(x)
        x = self.dropout(x)
        if self.residual:
            x = x + res
        return x


class ConvNeXtUpSampleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, bottleneck_channels=None,
                 stride=2, kernel_size=5,
                 norm='LN', dropout=0, residual=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=stride)
        self.convnext = ConvNeXtBlock1D(
            in_channels, out_channels, seq_len * stride, bottleneck_channels,
            1, kernel_size,
            norm, dropout, residual
        )

    def forward(self, x):
        """x: n, c, l"""
        x = self.upsample(x)
        x = self.convnext(x)
        return x


class UpSampleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len,
                 kernel_size=3,
                 stride=2,
                 norm='LN', dropout=0):
        """
        dcgan upsample block
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=stride)
        self.norm = parse_norm(in_channels, norm)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  padding=kernel_size // 2, stride=1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.upsample(x)
        x = self.conv_out(x)
        x = self.norm(x)
        x = F.leaky_relu(x, inplace=True)
        return x


class DownSampleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len,
                 kernel_size=3,
                 stride=2,
                 norm='LN', dropout=0):
        """
        dcgan downsample block
        """
        super().__init__()
        self.norm = parse_norm(in_channels, norm)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=kernel_size,
                                  padding=kernel_size // 2, stride=stride)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv_out(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.norm(x)
        return x


class CNNExtractor(nn.Module):
    def __init__(self, in_channels, seq_len,
                 stride_list=(2, 2, 2, 2), out_channels_list=(128, 128, 128, 128),
                 kernel_size_list=(5, 5, 5, 5),
                 norm='LN', input_norm=False):
        super().__init__()
        self.net = []
        if input_norm:
            self.norm_layer = parse_norm(in_channels, norm)
        current_in_channels = in_channels
        current_seq_len = seq_len
        for stride, out_channels, kernel_size in zip(stride_list, out_channels_list, kernel_size_list):
            self.net.append(
                ConvNeXtBlock1D(
                    current_in_channels,
                    out_channels,
                    current_seq_len,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm
                )
            )
            current_in_channels = out_channels
            current_seq_len = current_seq_len // stride
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.norm_layer(x)
        return self.net(x)


class DecoderUpsample(nn.Module):
    def __init__(self, in_channels, seq_len,
                 stride_list=(2, 2, 2, 2),
                 out_channels_list=(128, 64, 32, 16),
                 kernel_size_list=(3, 3, 1, 1),
                 norm='LN',
                 dropout=0):
        super().__init__()
        self.net = []
        current_in_channels = in_channels
        current_seq_len = seq_len
        for stride, out_channels, kernel_size in \
                zip(stride_list, out_channels_list, kernel_size_list):
            self.net.append(
                ConvNeXtUpSampleBlock1D(
                    current_in_channels,
                    out_channels,
                    current_seq_len,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm,
                    dropout=dropout,
                )
                # UpSampleBlock1D(
                #     current_in_channels,
                #     out_channels,
                #     current_seq_len,
                #     stride=stride,
                #     kernel_size=kernel_size,
                #     norm=norm,
                #     dropout=dropout,
                # )
            )
            current_in_channels = out_channels
            current_seq_len = current_seq_len * stride
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """x: n, c, l"""
        return self.net(x)