import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
shrink layers
"""

class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec, batch_first=False):
        """

        """
        super(PosEncoding, self).__init__()
        self.batch_first = batch_first
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
             for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        self.pos_enc = nn.Embedding(max_seq_len, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, x):
        """x: n, l, c(batch_first=True) or l, n, c(batch_first=False)"""
        # -> n, l, c
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        input_pos = torch.arange(x.shape[1], device=x.device)

        enc = self.pos_enc(input_pos).float()
        x = enc + x
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x


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
                 stride=1, kernel_size=7,
                 norm='LN', dropout=0, residual=True):
        super().__init__()
        self.residual = residual
        if bottleneck_channels is None:
            bottleneck_channels = 2 * in_channels
        if norm is None:
            self.norm_bottleneck = nn.Identity()
        else:
            self.norm_bottleneck = nn.BatchNorm1d(in_channels) if norm == 'BN' else nn.LayerNorm(seq_len // stride)
        padding = kernel_size // 2
        self.conv_in = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv_bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, padding=0, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        """x: n, c, l"""
        res = x
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
                 stride=2, kernel_size=7,
                 norm='LN', dropout=0, residual=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=stride)
        self.convnext = ConvNeXtBlock1D(
            in_channels, out_channels, seq_len, bottleneck_channels,
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
        if norm is None:
            self.norm = nn.Identity()
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(in_channels)
        elif norm == 'LN':
            self.norm = nn.LayerNorm(seq_len)
        else:
            raise ValueError('unknown normalize')
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
        if norm is None:
            self.norm = nn.Identity()
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(in_channels)
        elif norm == 'LN':
            self.norm = nn.LayerNorm(seq_len)
        else:
            raise ValueError('unknown normalize')
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
                 kernel_size_list=(7, 7, 7, 7),
                 norm='LN'):
        super().__init__()
        self.net = []
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


class Generator(nn.Module):
    """
    use Unet like structure
    """

    def __init__(self, seq_len, tgt_dim, audio_feature_dim, noise_dim=16, norm='BN',
                 middle_dim=256, preprocess_dim=16, cls_label_dim=5):
        super(Generator, self).__init__()
        self.seq_len = seq_len

        self.tgt_dim = tgt_dim
        self.noise_dim = noise_dim
        self.cls_label_dim = cls_label_dim

        self.middle_dim = middle_dim

        self.preprocess = CNNExtractor(
            audio_feature_dim,
            seq_len,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        )

        self.noise_preprocess = CNNExtractor(
            noise_dim,
            seq_len,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        )

        self.cls_label_preprocess = CNNExtractor(
            cls_label_dim,
            seq_len,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        )

        self.body = CNNExtractor(
            preprocess_dim * 3,
            seq_len,
            stride_list=(2, 2, 2),
            out_channels_list=(middle_dim // 2, middle_dim // 2, middle_dim // 2),
            kernel_size_list=(7, 5, 5),
            norm=norm,
        )
        self.decoders = nn.ModuleList([nn.Sequential(
            DecoderUpsample(
                middle_dim // 2,
                seq_len // (2 ** 3),
                stride_list=(2, 2, 2),
                out_channels_list=(middle_dim // 4, middle_dim // 8, middle_dim // 16),
                kernel_size_list=(5, 3, 3),
                norm=norm,
            ),
            # to avoid last layer being leaky_relu
            nn.Conv1d(middle_dim // 16, 1, kernel_size=1),
            nn.Tanh(),
        ) for _ in range(self.tgt_dim)])

    def forward(self, cond_data):
        audio_feature, cls_label = cond_data
        N, L, _ = audio_feature.shape
        noise = torch.randn([N, self.noise_dim, L], device=audio_feature.device)
        x = self.preprocess(audio_feature.permute(0, 2, 1))
        cls_label_feature = self.cls_label_preprocess(
            # N, cls_label_dim
            F.one_hot(cls_label, num_classes=self.cls_label_dim).unsqueeze(dim=-1).expand(-1, -1, self.seq_len).permute(0, 2, 1)
        )
        # -> n, c, l
        x = torch.cat([x, noise, cls_label_feature], dim=1)
        x = self.body(x)
        output = [dec(x) for i, dec in enumerate(self.decoders)]
        # -> n, l, c
        return torch.cat(output, dim=1).permute(0, 2, 1)


class Discriminator(nn.Module):
    def __init__(self, seq_len, tgt_dim, audio_feature_dim, norm='LN', middle_dim=256,
                 preprocess_dim=16,
                 **kwargs):
        """
        output_feature_num = num_classes(density map) + 2(x, y)
        """
        super(Discriminator, self).__init__()
        self.tgt_dim = tgt_dim
        self.seq_len = seq_len

        self.preprocess = CNNExtractor(
            audio_feature_dim,
            seq_len,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        )

        self.tgt_preprocess = nn.ModuleList([CNNExtractor(
            1,
            seq_len,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        ) for _ in range(tgt_dim)])

        self.body = CNNExtractor(
            preprocess_dim * (self.tgt_dim + 1),
            seq_len,
            stride_list=(1, 2, 2, 2,
                         2, 2, 2),
            out_channels_list=(middle_dim // 16, middle_dim // 8, middle_dim // 4, middle_dim // 2,
                               middle_dim // 2, middle_dim // 2, middle_dim,
                               ),
            kernel_size_list=(1, 3, 3, 5,
                              5, 5, 7),
            norm=norm,
        )

        self.validity_head = nn.Linear(middle_dim, 1)
        self.cls_head = nn.Linear(middle_dim, 1)

    def forward(self, cond_data, tgt):
        """
        tgt: N, L, label_dim
        cond_data: N, L, cond_data_feature_dim
        """
        N, L, _ = cond_data.shape
        audio_feature = cond_data
        # N, L, _ = gen_output.shape
        x = self.preprocess(audio_feature.permute(0, 2, 1))

        x = torch.cat([x] + [
            self.tgt_preprocess[i](tgt[:, i])
            for i in range(self.tgt_dim)
        ], dim=1)
        # Concatenate label embedding and image to produce input
        # -> N, C, L
        x = self.body(x)

        # -> n, 1, l
        x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=-1)
        validity = self.validity_head(x)
        cls_logits = self.cls_head(x)

        # N, 1
        return validity, cls_logits
