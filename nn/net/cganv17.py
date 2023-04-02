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
            ConvNeXtBlock(
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


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, bottleneck_channels=None,
                 stride=1, kernel_size=7,
                 norm='LN', dropout=0):
        super().__init__()
        if bottleneck_channels is None:
            bottleneck_channels = 2 * in_channels
        if stride == 1:
            self.norm_in = nn.Identity()
        else:
            self.norm_in = nn.BatchNorm1d(in_channels) if norm == 'BN' else nn.LayerNorm(seq_len)
        padding = kernel_size // 2
        self.conv_in = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.norm_bottleneck = nn.BatchNorm1d(in_channels) if norm == 'BN' else nn.LayerNorm(seq_len // stride)
        self.conv_bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, padding=0, stride=1)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        """x: n, c, l"""
        # print('x.device')
        # print(x.device)
        x = self.norm_in(x)
        x = F.leaky_relu(self.conv_in(x), inplace=True)
        x = self.norm_bottleneck(x)
        x = F.leaky_relu(self.conv_bottleneck(x), inplace=True)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv_out(x))
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
                ConvNeXtBlock(
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


class UpConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, h, w,
                 kernel_size=3, stride=2, upconv_padding=0, upconv_out_padding=0,
                 out_conv_kernel_size=3,
                 norm='LN', dropout=0):
        """
        dim is seq_len after up_sampling
        """
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels,
                                        kernel_size=kernel_size,
                                        padding=upconv_padding,
                                        output_padding=upconv_out_padding,
                                        stride=stride)
        if norm is None:
            self.norm = nn.Identity()
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(in_channels)
        elif norm == 'LN':
            assert h is not None
            assert w is not None
            self.norm = nn.LayerNorm([h, w])
        else:
            raise ValueError('unknown normalize')
        self.dropout = nn.Dropout(dropout)
        self.conv_out = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=out_conv_kernel_size, padding=out_conv_kernel_size // 2, stride=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(self.norm(x)), inplace=True)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv_out(x))
        return x


class UpSampleBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len,
                 out_conv_kernel_size=3,
                 stride=2,
                 norm='LN', dropout=0):
        """
        dim is seq_len after up_sampling
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
                                  kernel_size=out_conv_kernel_size,
                                  padding=out_conv_kernel_size // 2, stride=1)

    def forward(self, x):
        x = F.leaky_relu(self.upsample(self.norm(x)), inplace=True)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv_out(x))
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, a,
                 stride_list=(2, 2, 2, 2),
                 out_channels_list=(128, 64, 32, 16),
                 kernel_size_list=(2, 2, 2, 2),
                 upconv_padding_list=(0, 0, 0, 0),
                 upconv_out_padding_list=(0, 0, 0, 0),
                 out_conv_kernel_size_list=(3, 3, 1, 1),
                 norm='LN',
                 dropout=0):
        super().__init__()
        self.net = []
        current_in_channels = in_channels
        current_a = a
        for stride, out_channels, kernel_size, out_conv_kernel_size, upconv_padding, upconv_out_padding in \
                zip(stride_list, out_channels_list, kernel_size_list, out_conv_kernel_size_list, upconv_padding_list,
                    upconv_out_padding_list):
            self.net.append(
                UpConvBlock2D(
                    current_in_channels,
                    out_channels,
                    current_a,
                    current_a,
                    kernel_size=kernel_size,
                    stride=stride,
                    upconv_padding=upconv_padding,
                    upconv_out_padding=upconv_out_padding,
                    out_conv_kernel_size=out_conv_kernel_size,
                    norm=norm,
                    dropout=dropout,
                )
            )
            current_in_channels = out_channels
            current_a = current_a * stride
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """x: n, c, l"""
        return self.net(x)


class DecoderUpsample(nn.Module):
    def __init__(self, in_channels, seq_len,
                 stride_list=(2, 2, 2, 2),
                 out_channels_list=(128, 64, 32, 16),
                 out_conv_kernel_size_list=(3, 3, 1, 1),
                 norm='LN',
                 dropout=0):
        super().__init__()
        self.net = []
        current_in_channels = in_channels
        current_seq_len = seq_len
        for stride, out_channels, out_conv_kernel_size in \
                zip(stride_list, out_channels_list, out_conv_kernel_size_list):
            self.net.append(
                UpSampleBlock1D(
                    current_in_channels,
                    out_channels,
                    current_seq_len,
                    stride=stride,
                    out_conv_kernel_size=out_conv_kernel_size,
                    norm=norm,
                    dropout=dropout,
                )
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

    def __init__(self, seq_len, label_dim, cond_data_feature_dim, noise_dim=16, norm='BN',
                 middle_dim=256, preprocess_dim=16):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.middle_dim = middle_dim
        self.label_dim = label_dim

        self.preprocess = CNNExtractor(
            cond_data_feature_dim,
            seq_len,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        )

        self.head = CNNExtractor(
            preprocess_dim + noise_dim,
            seq_len,
            stride_list=(2, 2, 2),
            out_channels_list=(middle_dim // 4, middle_dim // 2, middle_dim * label_dim // 2),
            kernel_size_list=(7, 5, 5),
            norm=norm,
        )
        # self.expansion = CNNExtractor(
        #     middle_dim,
        #     seq_len // (2 ** 3),
        #     stride_list=(1,),
        #     out_channels_list=(middle_dim * label_dim,),
        #     kernel_size_list=(1,),
        #     norm=norm,
        # )
        # self.decoders = nn.ModuleList([
        #     nn.Sequential(
        #         DecoderUpsample(
        #             middle_dim,
        #             seq_len // (2 ** 3),
        #             stride_list=(2, 2, 2),
        #             out_channels_list=(middle_dim, middle_dim // 4, middle_dim // 16),
        #             out_conv_kernel_size_list=(5, 3, 3),
        #             norm=norm,
        #         ),
        #         # to avoid last layer being leaky_relu
        #         nn.Conv1d(middle_dim // 16, 1, kernel_size=1)
        #     )
        #     for _ in range(label_dim)
        # ])
        self.decoders = nn.ModuleList([nn.Sequential(
            DecoderUpsample(
                middle_dim // 2,
                seq_len // (2 ** 3),
                stride_list=(2, 2, 2),
                out_channels_list=(middle_dim // 4, middle_dim // 8, middle_dim // 16),
                out_conv_kernel_size_list=(5, 3, 3),
                norm=norm,
            ),
            # to avoid last layer being leaky_relu
            nn.Conv1d(middle_dim // 16, 1, kernel_size=1)
        ) for _ in range(self.label_dim)])

    def forward(self, cond_data):
        N, L, _ = cond_data.shape
        noise = torch.randn([N, self.noise_dim, L], device=cond_data.device)
        x = self.preprocess(cond_data.permute(0, 2, 1))
        # -> n, c, l
        x = torch.cat([x, noise], dim=1)
        x = self.head(x)
        # x = self.stem(x)
        N, C, L = x.shape
        # output = self.decoders(x)
        x = x.reshape([N, self.label_dim, -1, L])
        output = [dec(x[:, i]) for i, dec in enumerate(self.decoders)]
        # -> n, l, c
        return torch.cat(output, dim=1).permute(0, 2, 1)


class Discriminator(nn.Module):
    def __init__(self, seq_len, label_dim, cond_data_feature_dim, norm='LN', middle_dim=256,
                 preprocess_dim=16,
                 **kwargs):
        """
        output_feature_num = num_classes(density map) + 2(x, y)
        """
        super(Discriminator, self).__init__()
        self.label_dim = label_dim
        self.seq_len = seq_len

        self.preprocess = CNNExtractor(
            cond_data_feature_dim,
            seq_len,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        )
        self.expand_output = CNNExtractor(
            label_dim,
            seq_len,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        )
        self.head = CNNExtractor(
            preprocess_dim * 2,
            seq_len,
            stride_list=(2, 2, 2, 2, 2, 2),
            out_channels_list=(middle_dim // 8, middle_dim // 4,
                               middle_dim // 4, middle_dim // 2,
                               middle_dim, middle_dim,),
            kernel_size_list=(7, 7, 5, 5, 3, 3),
            norm=norm,
        )
        self.fcn = nn.Linear(middle_dim, 1)

    def forward(self, cond_data, gen_output):
        """
        gen_output: N, L, label_dim
        cond_data: N, L, cond_data_feature_dim
        """
        N, L, _ = cond_data.shape
        # N, L, _ = gen_output.shape
        x = self.preprocess(cond_data.permute(0, 2, 1))

        gen_output = self.expand_output(gen_output.permute(0, 2, 1))
        # Concatenate label embedding and image to produce input
        # -> N, C, L
        x = self.head(torch.cat([x, gen_output], dim=1))

        # x = self.stem(x)

        # -> n, 1, l
        x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=-1)
        validity = self.fcn(x)

        # N, 1
        return validity
