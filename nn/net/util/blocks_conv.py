from nn.net.util.transformer_modules import PosEncoding
from torch import nn
from torch.nn import functional as F
import torch


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
            ConvBlock1DNoBottleneck(
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


class ConvBlock1DNoBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,
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
        self.norm_in = parse_norm(in_channels, norm)
        self.norm_out = parse_norm(out_channels, norm)
        padding = kernel_size // 2
        self.conv_in = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv_out = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1)

    def forward(self, x):
        """x: n, c, l"""
        res = self.short_cut(x)
        x = self.conv_in(x)
        x = self.norm_in(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.conv_out(x)
        x = self.norm_out(x)
        x = F.leaky_relu(x, inplace=True)
        if self.residual:
            x = x + res
        return x


class ConvUpSampleBlock1DNoBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=2, kernel_size=5,
                 norm='LN', dropout=0, residual=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=stride) if stride != 1 else nn.Identity()
        self.convnext = ConvBlock1DNoBottleneck(
            in_channels, out_channels,
            1, kernel_size,
            norm, dropout, residual
        )

    def forward(self, x):
        """x: n, c, l"""
        x = self.upsample(x)
        x = self.convnext(x)
        return x


class ConvDownSampleBlock1DNoBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=2, kernel_size=5,
                 norm='LN', dropout=0, residual=True):
        super().__init__()
        self.stride = stride
        self.convnext = ConvBlock1DNoBottleneck(
            in_channels, out_channels,
            1, kernel_size,
            norm, dropout, residual
        )

    def forward(self, x):
        """x: n, c, l"""
        x = self.convnext(x)
        N, C, L = x.shape
        x = torch.mean(x.reshape(N, C, L // self.stride, self.stride), dim=-1)
        return x


class CNNExtractor(nn.Module):
    def __init__(self, in_channels,
                 stride_list=(2, 2, 2, 2), out_channels_list=(128, 128, 128, 128),
                 kernel_size_list=(5, 5, 5, 5),
                 norm='LN', input_norm=False, first_layer_residual=True):
        super().__init__()
        self.net = []
        self.norm = norm
        if input_norm:
            self.norm_layer = parse_norm(in_channels, norm)
        current_in_channels = in_channels
        residual_list = [first_layer_residual] + [True] * (len(stride_list) - 1)
        for stride, out_channels, kernel_size, residual in zip(
                stride_list, out_channels_list, kernel_size_list, residual_list):
            self.net.append(
                ConvBlock1DNoBottleneck(
                    current_in_channels,
                    out_channels,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm,
                    residual=residual,
                )
            )
            current_in_channels = out_channels
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.norm_layer(x)
        return self.net(x)


class CNNExtractorWithShortcut(nn.Module):
    def __init__(self, in_channels,
                 stride_list=(2, 2, 2, 2), out_channels_list=(128, 128, 128, 128),
                 kernel_size_list=(5, 5, 5, 5),
                 shortcut_start_pos_list=tuple(),
                 aggregate='mean',
                 norm='LN', input_norm=False, first_layer_residual=True):
        super().__init__()
        self.shortcut_start_pos_list = shortcut_start_pos_list
        self.aggregate = aggregate
        self.net = []
        self.norm = norm
        if input_norm:
            if input_norm:
                self.norm_layer = parse_norm(in_channels, norm)
        current_in_channels = in_channels
        residual_list = [first_layer_residual] + [True] * (len(stride_list) - 1)
        for stride, out_channels, kernel_size, residual in zip(
                stride_list, out_channels_list, kernel_size_list, residual_list):
            self.net.append(
                ConvBlock1DNoBottleneck(
                    current_in_channels,
                    out_channels,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm,
                    residual=residual,
                )
            )
            current_in_channels = out_channels
        self.net = nn.ModuleList(self.net)

    def forward(self, x):
        x = self.norm_layer(x)
        shortcut_x = []
        for i, layer in enumerate(self.net):
            if i in self.shortcut_start_pos_list:
                shortcut_x.append(x)
            x = layer(x)
        shortcut_x.append(x)
        if self.aggregate == 'mean':
            return torch.mean(torch.stack(shortcut_x, dim=0), dim=0)
        else:
            return torch.cat(shortcut_x, dim=1)


class CNNExtractorDownsample(nn.Module):
    def __init__(self, in_channels,
                 stride_list=(2, 2, 2, 2), out_channels_list=(128, 128, 128, 128),
                 kernel_size_list=(5, 5, 5, 5),
                 norm='LN', input_norm=False, first_layer_residual=True):
        super().__init__()
        self.net = []
        self.norm = norm
        if input_norm:
            self.norm_layer = parse_norm(in_channels, norm)
        current_in_channels = in_channels
        residual_list = [first_layer_residual] + [True] * (len(stride_list) - 1)
        for stride, out_channels, kernel_size, residual in zip(
                stride_list, out_channels_list, kernel_size_list, residual_list):
            self.net.append(
                ConvBlock1DNoBottleneck(
                    current_in_channels,
                    out_channels,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm,
                    residual=residual,
                )
            )
            current_in_channels = out_channels
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        x = self.norm_layer(x)
        return self.net(x)


class DecoderUpsample(nn.Module):
    def __init__(self, in_channels,
                 stride_list=(2, 2, 2, 2),
                 out_channels_list=(128, 64, 32, 16),
                 kernel_size_list=(3, 3, 1, 1),
                 norm='LN',
                 first_layer_residual=True,
                 dropout=0):
        super().__init__()
        self.net = []
        current_in_channels = in_channels
        residual_list = [first_layer_residual] + [True] * (len(stride_list) - 1)
        for stride, out_channels, kernel_size, residual in zip(
                stride_list, out_channels_list, kernel_size_list, residual_list):
            self.net.append(
                ConvUpSampleBlock1DNoBottleneck(
                    current_in_channels,
                    out_channels,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm,
                    dropout=dropout,
                    residual=residual,
                )
            )
            current_in_channels = out_channels
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """x: n, c, l"""
        return self.net(x)
