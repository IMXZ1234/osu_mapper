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
        if (not self.residual) or ((in_channels == out_channels) and (stride == 1)):
            self.short_cut = nn.Identity()
        else:
            if norm is None:
                norm_short_cut = nn.Identity()
            else:
                norm_short_cut = nn.BatchNorm1d(out_channels) if norm == 'BN' else nn.LayerNorm(seq_len // stride)
            self.short_cut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
                norm_short_cut
            )
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
                 stride=2, kernel_size=7,
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
                 norm='LN', input_norm=False):
        super().__init__()
        self.net = []
        if input_norm:
            if norm is None:
                pass
            elif norm == 'BN':
                self.net.append(nn.BatchNorm1d(in_channels))
            elif norm == 'LN':
                self.net.append(nn.LayerNorm(seq_len))
            else:
                raise ValueError('unknown normalize')
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
    we predict 16 dim hit_signal embedding + 2 dim x&y coordinate
    """

    def __init__(self, n_snap, tgt_embedding_dim=16, tgt_coord_dim=2, audio_feature_dim=40, noise_dim=16, norm='BN',
                 middle_dim=256, preprocess_dim=32, condition_coeff=1.,
                 # one-hot acgan predicted label (beatmap difficulty here)
                 cls_label_dim=5):
        super(Generator, self).__init__()
        # 16 mel frames per snap, 8 snaps per beat, 1 hit_signal embedding label per beat
        n_frames = n_snap * 16
        self.n_frames = n_frames
        middle_seq_len = n_frames // (2 ** 4)
        self.middle_seq_len = middle_seq_len

        self.tgt_embedding_dim = tgt_embedding_dim
        self.tgt_coord_dim = tgt_coord_dim
        self.noise_dim = noise_dim
        self.cls_label_dim = cls_label_dim

        self.condition_coeff = condition_coeff

        self.middle_dim = middle_dim

        self.preprocess = CNNExtractor(
            audio_feature_dim,
            n_frames,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
            input_norm=True,
        )

        self.noise_preprocess = CNNExtractor(
            noise_dim,
            middle_seq_len,
            stride_list=(1,),
            out_channels_list=(middle_dim,),
            kernel_size_list=(1,),
            norm=norm,
            input_norm=True,
        )

        self.cls_label_preprocess = CNNExtractor(
            cls_label_dim,
            n_frames,
            stride_list=(1,),
            out_channels_list=(preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
            input_norm=True,
        )

        # shrink 2 ** 4, equal to length of embedding final output
        self.body = CNNExtractor(
            preprocess_dim * 2,
            n_frames,
            stride_list=(2, 2, 2, 2),
            out_channels_list=(middle_dim // 4, middle_dim // 2, middle_dim, middle_dim),
            kernel_size_list=(7, 7, 7, 7),
            norm=norm,
        )
        self.condition_norm = nn.LayerNorm(middle_seq_len)

        # generate x & y coord for each snap
        self.decoder_coord = nn.Sequential(
            CNNExtractor(
                middle_dim,
                middle_seq_len,
                stride_list=(1, 1, 1, 1),
                out_channels_list=(middle_dim, middle_dim // 2,
                                   middle_dim // 4, middle_dim // 8),
                kernel_size_list=(5, 5, 3, 3),
                norm=norm,
            ),
            # to avoid last layer being leaky_relu
            nn.Conv1d(middle_dim // 8, self.tgt_coord_dim, kernel_size=1),
            # nn.Sigmoid(),
        )

        # downsample 2 ** 3, generate x & y coord for each snap
        self.decoder_embedding = nn.Sequential(
            CNNExtractor(
                middle_dim,
                middle_seq_len,
                stride_list=(2, 2, 2, 1),
                out_channels_list=(middle_dim, middle_dim // 2,
                                   middle_dim // 4, middle_dim // 8),
                kernel_size_list=(5, 5, 3, 3),
                norm=norm,
            ),
            # to avoid last layer being leaky_relu
            nn.Conv1d(middle_dim // 8, self.tgt_embedding_dim, kernel_size=1),
            # nn.Sigmoid(),
        )

    def forward(self, cond_data, noise=None):
        audio_feature, cls_label = cond_data
        N, L, _ = audio_feature.shape
        assert L == self.n_frames
        x = self.preprocess(audio_feature.permute(0, 2, 1))
        cls_label_feature = self.cls_label_preprocess(
            # N, cls_label_dim
            cls_label.unsqueeze(dim=-1).expand(-1, -1, self.n_frames)
        )
        # -> n, c, l
        x = torch.cat([x, cls_label_feature], dim=1)
        x = self.body(x)

        if noise is None:
            noise = torch.randn([N, self.noise_dim, self.middle_seq_len], device=audio_feature.device)
        noise = self.noise_preprocess(noise)
        x = self.condition_norm(x) * noise

        coord_output = self.decoder_coord(x)
        embed_output = self.decoder_embedding(x)
        # -> n, l, c
        return coord_output.permute(0, 2, 1), embed_output.permute(0, 2, 1)


class Discriminator(nn.Module):
    def __init__(self, n_snap, tgt_embedding_dim=16, tgt_coord_dim=2, audio_feature_dim=40, norm='LN', middle_dim=256,
                 preprocess_dim=32, cls_label_dim=5, validity_sigmoid=False,
                 **kwargs):
        """
        output_feature_num = num_classes(density map) + 2(x, y)
        """
        super(Discriminator, self).__init__()
        # 16 mel frames per snap, 8 snaps per beat, 1 hit_signal embedding label per beat
        n_frames = n_snap * 16
        self.n_frames = n_frames
        n_beats = n_snap // 8
        self.n_beats = n_beats

        self.tgt_embedding_dim = tgt_embedding_dim
        self.tgt_coord_dim = tgt_coord_dim
        self.validity_sigmoid = validity_sigmoid

        # mel frame input, downsample 2 ** 7
        self.preprocess = CNNExtractor(
            audio_feature_dim,
            n_frames,
            stride_list=(2, 4, 4, 4),
            out_channels_list=(preprocess_dim, preprocess_dim, preprocess_dim, preprocess_dim),
            kernel_size_list=(7, 7, 7, 7),
            norm=norm,
        )

        # downsample 2 ** 3
        self.tgt_coord_preprocess = CNNExtractor(
            self.tgt_coord_dim,
            n_snap,
            stride_list=(2, 2, 2, 1),
            out_channels_list=(preprocess_dim, preprocess_dim, preprocess_dim, preprocess_dim),
            kernel_size_list=(7, 7, 7, 7),
            norm=norm,
            input_norm=True,
        )

        self.tgt_embedding_preprocess = CNNExtractor(
            self.tgt_embedding_dim,
            n_beats,
            stride_list=(1, 1, 1, 1),
            out_channels_list=(preprocess_dim, preprocess_dim, preprocess_dim, preprocess_dim),
            kernel_size_list=(7, 7, 7, 7),
            norm=norm,
            input_norm=True,
        )

        self.body = CNNExtractor(
            preprocess_dim * 3,
            n_beats,
            stride_list=(1, 1, 1, 1, 1),
            out_channels_list=(middle_dim, middle_dim, middle_dim, middle_dim, middle_dim),
            kernel_size_list=(5, 5, 5, 3, 3),
            norm=norm,
        )

        self.validity_head = nn.Sequential(
            nn.Linear(middle_dim, 1),
            nn.Sigmoid() if self.validity_sigmoid else nn.Identity(),
        )
        self.cls_head = nn.Linear(middle_dim, cls_label_dim)

    def forward(self, cond_data, tgt):
        """
        tgt: N, L, tgt_dim
        cond_data: N, L, audio_feature_dim
        """
        N, L, _ = cond_data.shape
        audio_feature = cond_data
        # N, L, _ = gen_output.shape -> N, C, L
        x = self.preprocess(audio_feature.permute(0, 2, 1))
        # N, L, C -> N, C, L
        coord_output, embed_output = tgt
        coord_output = coord_output.permute(0, 2, 1)
        embed_output = embed_output.permute(0, 2, 1)
        # print('tgt.shape', tgt.shape)
        x = torch.cat([
            x,
            self.tgt_coord_preprocess(coord_output),
            self.tgt_embedding_preprocess(embed_output)
        ], dim=1)
        # Concatenate label embedding and image to produce input
        # -> N, C, L
        x = self.body(x)

        # -> n, 1, l
        x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=-1)
        validity = self.validity_head(x)
        cls_logits = self.cls_head(x)
        # cls_logits = torch.sigmoid(self.cls_head(x))

        # N, 1
        return validity, cls_logits


if __name__ == '__main__':
    import thop

    subseq_len = 2560
    snap_feature = 41
    gen_params = {
        'seq_len': subseq_len,
        'tgt_dim': 5,
        'noise_dim': 16,
        'audio_feature_dim': snap_feature,
        'norm': 'LN',
        'middle_dim': 128,
        'preprocess_dim': 16,
        'cls_label_dim': 4,
    }
    dis_params = {
        'seq_len': subseq_len,
        'tgt_dim': 5,
        # 'noise_dim': 64,
        'audio_feature_dim': snap_feature,
        'norm': 'LN',
        'middle_dim': 128,
        'preprocess_dim': 16,
        'cls_label_dim': 4,
    }
    gen = Generator(**gen_params)
    dis = Discriminator(**dis_params)

    gen_input = ((torch.zeros([1, 2560, 41]), torch.zeros([1, 4])),)
    dis_input = (torch.zeros([1, 2560, 41]), torch.zeros([1, 2560, 5]))
    print(thop.profile(gen, gen_input))
    print(thop.profile(dis, dis_input))
