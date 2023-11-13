import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.net.util.blocks_conv_no_bottleneck import CNNExtractor, DecoderUpsample



# class AudioHead(nn.Module):
#     def __init__(self, ):
#         super(AudioHead, self).__init__()


class Generator(nn.Module):
    """
    we predict 16 dim hit_signal embedding + 2 dim x&y coordinate
    """

    def __init__(self, n_snap, tgt_embedding_dim=16, tgt_coord_dim=2, audio_feature_dim=40, noise_dim=256, norm='BN',
                 middle_dim=256, audio_preprocess_dim=24, label_preprocess_dim=1, condition_coeff=1.,
                 # one-hot acgan predicted label (beatmap difficulty here)
                 cls_label_dim=3):
        super(Generator, self).__init__()
        # 16 mel frames per snap, 8 snaps per beat, 1 hit_signal embedding label per beat
        self.n_snaps = n_snap
        n_beats = self.n_snaps // 8
        self.n_beats = n_beats
        n_frames = n_snap * 16
        self.n_frames = n_frames

        self.tgt_embedding_dim = tgt_embedding_dim
        self.tgt_coord_dim = tgt_coord_dim
        self.noise_dim = noise_dim
        self.cls_label_dim = cls_label_dim

        self.condition_coeff = condition_coeff

        self.middle_dim = middle_dim
        self.init_noise_dim = self.noise_dim // 16
        self.noise_init = nn.Linear(self.noise_dim, self.init_noise_dim * n_beats)

        # downsample 16 * 8
        self.preprocess_audio = CNNExtractor(
            audio_feature_dim,
            n_frames,
            stride_list=(8, 4, 4),
            out_channels_list=(audio_feature_dim, audio_feature_dim, audio_preprocess_dim,),
            kernel_size_list=(15, 7, 7),
            norm=norm,
        )

        self.noise_preprocess = DecoderUpsample(
            self.init_noise_dim,
            n_beats,
            stride_list=(1, 1, 1),
            out_channels_list=(noise_dim, noise_dim, noise_dim),
            kernel_size_list=(3, 3, 3),
            norm=norm,
        )

        self.cls_label_preprocess = CNNExtractor(
            cls_label_dim,
            n_beats,
            stride_list=(1,),
            out_channels_list=(label_preprocess_dim,),
            kernel_size_list=(1,),
            norm=norm,
        )

        self.switch = nn.Sequential(
            nn.Conv1d(noise_dim + audio_preprocess_dim + label_preprocess_dim, middle_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(middle_dim) if norm == 'BN' else nn.LayerNorm(n_beats),
            nn.LeakyReLU(inplace=True),
        )

        # keep dim
        self.body = CNNExtractor(
            middle_dim,
            n_beats,
            stride_list=(1, 1, 1),
            out_channels_list=(middle_dim, middle_dim, middle_dim),
            kernel_size_list=(3, 3, 3),
            norm=norm,
        )
        # self.condition_norm = nn.LayerNorm(n_beats)

        # generate x & y coord for each snap
        self.decoder_coord = nn.Sequential(
            DecoderUpsample(
                middle_dim,
                n_beats,
                stride_list=(2, 2, 2, 1),
                out_channels_list=(middle_dim, middle_dim, middle_dim, middle_dim),
                kernel_size_list=(3, 3, 3, 3),
                norm=norm,
            ),
            # to avoid last layer being leaky_relu
            nn.Conv1d(middle_dim, self.tgt_coord_dim, kernel_size=1),
            # nn.Tanh(),
        )

        self.decoder_embedding = nn.Sequential(
            CNNExtractor(
                middle_dim,
                n_beats,
                stride_list=(1, 1, 1, 1),
                out_channels_list=(middle_dim, middle_dim, middle_dim, middle_dim),
                kernel_size_list=(3, 3, 3, 3),
                norm=norm,
            ),
            # to avoid last layer being leaky_relu
            nn.Conv1d(middle_dim, self.tgt_embedding_dim, kernel_size=1),
            # nn.Sigmoid(),
        )

    def forward(self, cond_data, noise=None):
        audio_feature, cls_label = cond_data
        N, L, C = audio_feature.shape
        assert L == self.n_frames
        audio = self.preprocess_audio(audio_feature.permute(0, 2, 1))
        cls_label_feature = self.cls_label_preprocess(
            # N, cls_label_dim
            cls_label.unsqueeze(dim=-1).expand(-1, -1, self.n_beats)
        )

        if noise is None:
            noise = torch.randn([N, self.noise_dim], device=audio_feature.device)
        noise = self.noise_init(noise).reshape([N, self.init_noise_dim, self.n_beats])
        noise = self.noise_preprocess(noise)
        # -> n, c, l
        x = torch.cat([audio, noise, cls_label_feature], dim=1)
        x = self.switch(x)
        x = self.body(x) + x

        coord_output = self.decoder_coord(x)
        embed_output = self.decoder_embedding(x)
        # -> n, l, c
        return coord_output.permute(0, 2, 1), embed_output.permute(0, 2, 1)


class Discriminator(nn.Module):
    def __init__(self, n_snap, tgt_embedding_dim=16, tgt_coord_dim=2, audio_feature_dim=40, norm='LN', middle_dim=256, audio_preprocess_dim=24,
                 cls_label_dim=5, validity_sigmoid=False,
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

        preprocess_dim = middle_dim // 2

        # mel frame input, downsample 2 ** 5
        self.preprocess = nn.Sequential(
            CNNExtractor(
                audio_feature_dim,
                n_frames,
                stride_list=(8, 4, 4),
                out_channels_list=(audio_feature_dim, audio_feature_dim, audio_preprocess_dim,),
                kernel_size_list=(15, 7, 7),
                norm=norm,
            )
        )

        # downsample 2 ** 3
        self.tgt_coord_preprocess = nn.Sequential(
            CNNExtractor(
                self.tgt_coord_dim,
                # middle_dim,
                n_snap,
                stride_list=(2, 2, 2),
                out_channels_list=(middle_dim, middle_dim, preprocess_dim),
                kernel_size_list=(3, 3, 3),
                norm=norm,
            )
        )

        # self.coord_shortcut = nn.Sequential(
        #     nn.Conv1d(self.tgt_coord_dim, middle_dim, kernel_size=9, padding=4),
        #     nn.LeakyReLU(0.02),
        # )
        self.coord_shortcut = nn.Sequential(
            nn.Conv1d(self.tgt_coord_dim, preprocess_dim, kernel_size=9, padding=4),
            nn.LeakyReLU(inplace=True),
        )

        self.tgt_embedding_preprocess = nn.Sequential(
            CNNExtractor(
                self.tgt_embedding_dim,
                # middle_dim,
                n_beats,
                stride_list=(1, 1, 1),
                out_channels_list=(middle_dim, middle_dim, preprocess_dim),
                kernel_size_list=(3, 3, 3),
                norm=norm,
            )
        )

        self.body = CNNExtractor(
            preprocess_dim * 2 + audio_preprocess_dim,
            # (middle_dim // 2) * 3,
            n_beats,
            stride_list=(1, 1, 1, 1),
            out_channels_list=(middle_dim, middle_dim, middle_dim, middle_dim),
            kernel_size_list=(3, 3, 3, 3),
            norm=norm,
        )

        self.validity_head = nn.Sequential(
            nn.Linear(middle_dim + preprocess_dim * 3, 1),
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
        audio_preprocess_out = self.preprocess(audio_feature.permute(0, 2, 1))
        # N, L, C -> N, C, L
        coord_output, embed_output = tgt
        coord_output = coord_output.permute(0, 2, 1)
        embed_output = embed_output.permute(0, 2, 1)
        coord_preprocess_out = self.tgt_coord_preprocess(coord_output)
        embedding_preprocess_out = self.tgt_embedding_preprocess(embed_output)
        from_preprocess = torch.cat([coord_preprocess_out, embedding_preprocess_out], dim=1)
        x = torch.cat([
            from_preprocess,
            audio_preprocess_out,
        ], dim=1)
        # Concatenate label embedding and image to produce input
        # -> N, C, L
        x = self.body(x)

        # -> n, c, 1
        x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=-1)

        from_shortcut = self.coord_shortcut(coord_output)
        from_shortcut = F.adaptive_avg_pool1d(from_shortcut, 1).squeeze(dim=-1)
        from_preprocess = F.adaptive_avg_pool1d(from_preprocess, 1).squeeze(dim=-1)

        v_x = torch.cat([x, from_shortcut, from_preprocess], dim=-1)

        validity = self.validity_head(v_x)
        cls_logits = self.cls_head(x)
        # cls_logits = torch.sigmoid(self.cls_head(x))

        # N, 1
        return validity, cls_logits


if __name__ == '__main__':
    import thop

    subseq_len = 2360
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
