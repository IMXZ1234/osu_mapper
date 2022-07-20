import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_feat, out_feat, kernel_size=3, normalize=True, act=True):
    layers = [nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, padding=kernel_size // 2)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat))
    if act:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class FeatureExtractor(nn.Module):
    """
    cond_data: N, L(output_len), snap_feature(514 here) -> N, L, out_feature
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.model = nn.Sequential(
            *conv_block(in_channels, in_channels // 4, normalize=False),
            *conv_block(in_channels // 4, in_channels // 16),
            *conv_block(in_channels // 16, out_channels, act=False),
        )

    def forward(self, cond_data):
        cond_data = cond_data.transpose(1, 2)
        cond_data = self.model(cond_data)
        return cond_data


class Generator(nn.Module):
    def __init__(self, output_len, in_channels, num_classes, noise_in_channels,
                 compressed_channels=16, **kwargs):
        """
        output_snaps = output_feature_num // num_classes
        """
        super(Generator, self).__init__()

        self.num_classes = num_classes
        self.output_len = output_len
        self.ext_in_channels = in_channels
        self.ext_out_channels = compressed_channels

        output_feature_num = self.output_len * self.num_classes

        # def block(in_feat, out_feat, normalize=True):
        #     layers = [nn.Linear(in_feat, out_feat)]
        #     # if normalize:
        #     #     layers.append(nn.BatchNorm1d(out_feat, 0.8))
        #     layers.append(nn.LeakyReLU(0.2, inplace=True))
        #     return layers
        self.ext = FeatureExtractor(self.ext_in_channels, self.ext_out_channels)
        print('G first linear in')
        print(noise_in_channels + self.ext_out_channels * self.output_len)
        print('G first linear out')
        print(16 * output_feature_num)

        self.model = nn.Sequential(
            *conv_block(noise_in_channels + self.ext_out_channels, self.num_classes * 32, kernel_size=7, normalize=False),
            *conv_block(self.num_classes * 32, self.num_classes * 16, kernel_size=7),
            *conv_block(self.num_classes * 16, self.num_classes * 16, kernel_size=5),
            *conv_block(self.num_classes * 16, self.num_classes * 4, kernel_size=5),
            *conv_block(self.num_classes * 4, self.num_classes * 4, kernel_size=3),
            *conv_block(self.num_classes * 4, self.num_classes, kernel_size=3, act=False),
        )

    def forward(self, noise, cond_data):
        # print('in G')
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('noise.shape')
        # print(noise.shape)
        # Concatenate label embedding and image to produce input
        cond_data = self.ext(cond_data)
        noise = noise.transpose(1, 2)
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('noise.shape')
        # print(noise.shape)
        gen_input = torch.cat((cond_data, noise), dim=1)
        # N, num_classes, num_snaps -> N, num_snaps, num_classes
        gen_output = self.model(gen_input).transpose(1, 2)
        gen_output = torch.nn.functional.gumbel_softmax(gen_output, dim=-1, hard=True)
        # N, num_snaps, num_classes
        # print('gen_output.shape')
        # print(gen_output.shape)
        return gen_output, cond_data


class Discriminator(nn.Module):
    def __init__(self, output_len, in_channels, num_classes,
                 compressed_channels=16, **kwargs):
        """
        output_feature_num = output_snaps * num_classes: generator output feature num, in one-hot format
        """
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.output_len = output_len
        self.ext_in_channels = in_channels
        self.ext_out_channels = compressed_channels

        self.model = nn.Sequential(
            *conv_block(self.num_classes + self.ext_out_channels, self.num_classes * 8, normalize=False),
            # *conv_block(self.num_classes * 8, self.num_classes * 8),
            *conv_block(self.num_classes * 8, self.num_classes * 2, act=False),
        )

        self.fcn = nn.Linear(self.num_classes * 2, 2)

    def forward(self, gen_output, cond_data):
        # print('in D')
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('gen_output.shape')
        # print(gen_output.shape)
        gen_output = gen_output.transpose(1, 2)
        # cond_data = self.ext(cond_data)
        # Concatenate label embedding and image to produce input
        d_in = torch.cat([gen_output, cond_data], dim=1)
        validity = self.model(d_in)
        validity = F.avg_pool1d(validity, validity.shape[2]).reshape([validity.shape[0], -1])

        validity = self.fcn(validity)

        # N, 2
        return validity
