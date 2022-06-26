import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    cond_data: N, L(output_len), snap_feature(514 here) -> N, L, out_feature -> N, -1
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        def conv_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv1d(in_feat, out_feat, kernel_size=3, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels, in_channels // 4, normalize=False),
            *conv_block(in_channels // 4, in_channels // 16),
            nn.Conv1d(in_channels // 16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, cond_data):
        cond_data = cond_data.transpose(1, 2)
        cond_data = self.model(cond_data)
        return cond_data.reshape([cond_data.shape[0], -1])


class Generator(nn.Module):
    def __init__(self, output_len, in_channels, num_classes, noise_feature_num,
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
        print(noise_feature_num + self.ext_out_channels * self.output_len)
        print('G first linear out')
        print(16 * output_feature_num)

        self.model = nn.Sequential(
            nn.Linear(noise_feature_num + self.ext_out_channels * self.output_len, 16 * output_feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16 * output_feature_num, 8 * output_feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8 * output_feature_num, 2 * output_feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2 * output_feature_num, output_feature_num),
        )

    def forward(self, noise, cond_data):
        # print('in G')
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('noise.shape')
        # print(noise.shape)
        # Concatenate label embedding and image to produce input
        cond_data = self.ext(cond_data)
        # print('cond_data.shape')
        # print(cond_data.shape)
        noise = noise.reshape([noise.shape[0], -1])
        # print('noise.shape')
        # print(noise.shape)
        gen_input = torch.cat((cond_data, noise), dim=-1)
        gen_output = self.model(gen_input)
        N, C = gen_output.shape  # N, output_feature_num
        gen_output = gen_output.reshape([N, C//self.num_classes, self.num_classes])
        # gen_output = torch.argmax(torch.softmax(gen_output, dim=-1), dim=-1)
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

        output_feature_num = self.output_len * self.num_classes

        # self.ext = FeatureExtractor(self.ext_in_channels, self.ext_out_channels)

        self.model = nn.Sequential(
            nn.Linear(self.ext_out_channels * self.output_len + output_feature_num, 4 * output_feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(4 * output_feature_num, 2 * output_feature_num),
            # nn.Dropout(0.4),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4 * output_feature_num, output_feature_num),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(output_feature_num, 2),
        )

    def forward(self, gen_output, cond_data):
        # print('in D')
        # print('cond_data.shape')
        # print(cond_data.shape)
        # print('gen_output.shape')
        # print(gen_output.shape)
        # cond_data = self.ext(cond_data)
        gen_output = gen_output.reshape([gen_output.shape[0], -1])
        # gen_output = torch.nn.functional.one_hot(gen_output)
        # Concatenate label embedding and image to produce input
        d_in = torch.cat([gen_output, cond_data], dim=-1)
        validity = self.model(d_in)
        # N, 2
        return validity
