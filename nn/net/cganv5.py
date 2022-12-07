import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block1d(in_feat, out_feat, kernel_size=(3), normalize=True, act=True):
    layers = [nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, padding=kernel_size // 2)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    if act:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


def conv_block2d(in_feat, out_feat, kernel_size=(3, 3), normalize=True, act=True):
    layers = [nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, padding=[k // 2 for k in kernel_size])]
    if normalize:
        layers.append(nn.BatchNorm2d(out_feat))
    if act:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class ContextExtractor(nn.Module):
    """
    cond_data: N, L(output_len), snap_feature(514 here) -> N, L, out_feature
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            *conv_block2d(in_channels, out_channels, kernel_size=1, normalize=False, act=False),
            # *conv_block(in_channels // 4, in_channels // 16),
            # *conv_block(in_channels // 16, out_channels, act=False),
        )

    def forward(self, cond_data):
        cond_data = cond_data.transpose(1, 2)
        cond_data = self.model(cond_data)
        return cond_data


class Generator(nn.Module):
    def __init__(self, seq_len, label_dim, noise_dim,
                 cond_data_feature_dim,
                 fold_len=16, **kwargs):
        """
        output_snaps = output_feature_num // num_classes
        """
        super(Generator, self).__init__()

        self.label_dim = label_dim
        self.seq_len = seq_len
        self.noise_dim = noise_dim
        self.fold_len = fold_len

        self.output_feature_num = self.seq_len * self.label_dim

        self.model = nn.Sequential(
            *conv_block2d(noise_dim + cond_data_feature_dim, self.label_dim * 16, kernel_size=(7, 7), normalize=False),
            *conv_block2d(self.label_dim * 16, self.label_dim * 16, kernel_size=(7, 7), normalize=False),
            # *conv_block(self.num_classes * 16, self.num_classes * 16, kernel_size=7),
            # *conv_block(self.num_classes * 16, self.num_classes * 8, kernel_size=3),
            *conv_block2d(self.label_dim * 16, self.label_dim * 4, kernel_size=(5, 5)),
            *conv_block2d(self.label_dim * 4, self.label_dim * 4, kernel_size=(3, 3)),
            nn.Flatten(),
            nn.Linear(self.label_dim * 4 * self.seq_len, self.output_feature_num),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(self.num_classes * self.output_len, self.num_classes * self.output_len),
            # *conv_block(self.num_classes * 4, self.num_classes, kernel_size=1),
            # *conv_block(self.num_classes, self.num_classes, act=False),
        )

    def forward(self, cond_data):
        """
        noise: N, L, noise_dim
        cond_data: N, L, cond_data_feature_dim
        """
        N, L, _ = cond_data.shape
        noise = torch.randn([N, L, self.noise_dim], device=cond_data.device)
        # print('in G')
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('noise.shape')
        # print(noise.shape)
        # Concatenate label embedding and image to produce input
        # cond_data = self.ext(cond_data)
        # noise = noise.transpose(1, 2)
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('noise.shape')
        # print(noise.shape)
        gen_input = torch.cat([cond_data, noise], dim=2).permute(0, 2, 1).reshape([N, -1, L // self.fold_len, self.fold_len])
        # print('gen_input.shape')
        # print(gen_input.shape)
        # N, num_classes, num_snaps -> N, num_snaps, num_classes
        # gen_output = self.model(gen_input).transpose(1, 2)
        # N * num_classes * num_snaps -> N, num_snaps, num_classes
        gen_output = self.model(gen_input).reshape([N, self.seq_len, self.label_dim])
        # gen_output = torch.nn.functional.gumbel_softmax(gen_output, dim=-1, hard=True)
        # N, num_snaps, num_classes
        # print('gen_output.shape')
        # print(gen_output.shape)
        return gen_output


class Discriminator(nn.Module):
    def __init__(self, seq_len, label_dim, cond_data_feature_dim,
                 fold_len=16, **kwargs):
        """
        output_feature_num = num_classes(density map) + 2(x, y)
        """
        super(Discriminator, self).__init__()
        self.label_dim = label_dim
        self.seq_len = seq_len
        self.fold_len = fold_len
        # self.ext_in_channels = in_channels
        # self.ext_out_channels = in_channels

        # self.ext = FeatureExtractor(self.ext_in_channels, self.ext_out_channels)

        self.model = nn.Sequential(
            *conv_block2d(self.label_dim + cond_data_feature_dim, self.label_dim * 16, kernel_size=(7, 7), normalize=False),
            *conv_block2d(self.label_dim * 16, self.label_dim * 16, kernel_size=(5, 5)),
            *conv_block2d(self.label_dim * 16, self.label_dim * 8, kernel_size=(3, 3)),
        )

        self.fcn = nn.Linear(self.label_dim * 8, 1)

    def forward(self, cond_data, gen_output):
        """
        gen_output: N, L, label_dim
        cond_data: N, L, cond_data_feature_dim
        """
        N, L, _ = cond_data.shape
        # print('in D')
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('gen_output.shape')
        # print(gen_output.shape)
        # cond_data = self.ext(cond_data)
        # gen_output = gen_output.transpose(1, 2)
        # cond_data = self.ext(cond_data)
        # Concatenate label embedding and image to produce input
        # -> N, C, L
        d_in = torch.cat([gen_output, cond_data], dim=2).permute(0, 2, 1).reshape([N, -1, L // self.fold_len, self.fold_len])
        # print('d_in.shape')
        # print(d_in.shape)
        # -> N, C, fold_len, L // fold_len
        validity = self.model(d_in)
        validity = F.avg_pool2d(validity, (validity.shape[2], validity.shape[3])).squeeze(-1).squeeze(-1)

        validity = self.fcn(validity)

        # N, 1
        return validity
