import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv1d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, normalize='LN', dim=None):
        super(double_conv1d_bn, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        if normalize is None:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
        elif normalize == 'BN':
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
        elif normalize == 'LN':
            assert dim is not None
            self.bn1 = nn.LayerNorm(dim)
            self.bn2 = nn.LayerNorm(dim // strides)
        else:
            raise ValueError('unknown normalize')

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class deconv1d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2, normalize='LN', dim=None):
        """
        dim is seq_len after up_sampling
        """
        super(deconv1d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        if normalize is None:
            self.bn1 = nn.Identity()
        elif normalize == 'BN':
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif normalize == 'LN':
            assert dim is not None
            self.bn1 = nn.LayerNorm(dim)
        else:
            raise ValueError('unknown normalize')

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class Generator(nn.Module):
    """
    use Unet like structure
    """
    def __init__(self, seq_len, label_dim, cond_data_feature_dim, noise_dim, normalize='LN'):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        self.layer1_conv = double_conv1d_bn(cond_data_feature_dim + noise_dim, 128, kernel_size=7, dim=seq_len, normalize=normalize)
        self.layer2_conv = double_conv1d_bn(128, 32, kernel_size=7, dim=seq_len // 2, normalize=normalize)

        self.layer3_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 4, normalize=normalize)
        self.layer4_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 16, normalize=normalize)

        self.layer5_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 64, normalize=normalize)

        self.layer6_conv = double_conv1d_bn(64, 32, kernel_size=5, dim=seq_len // 16, normalize=normalize)
        self.layer7_conv = double_conv1d_bn(64, 32, kernel_size=5, dim=seq_len // 4, normalize=normalize)
        # self.layer8_conv = double_conv1d_bn(64, 32, kernel_size=3)
        # self.layer9_conv = double_conv1d_bn(16, 8)

        self.deconv1 = deconv1d_bn(32, 32, kernel_size=4, strides=4, dim=seq_len // 16, normalize=normalize)
        self.deconv2 = deconv1d_bn(32, 32, kernel_size=4, strides=4, dim=seq_len // 4, normalize=normalize)
        # self.deconv3 = deconv1d_bn(32, 32, kernel_size=4, strides=4)
        # self.deconv4 = deconv1d_bn(16, 8)

        self.layer9_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 4, normalize=normalize)
        self.fc = nn.Conv1d(32, label_dim, kernel_size=1)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, cond_data):
        N, L, _ = cond_data.shape
        noise = torch.randn([N, self.noise_dim, L], device=cond_data.device)
        x = torch.cat([cond_data.permute(0, 2, 1), noise], dim=1)

        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool1d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool1d(conv2, 2)  # length of pool2 = length of output = 1/4 * length of input = 384*4 / 4 = 384

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool1d(conv3, 4)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool1d(conv4, 4)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        conv9 = self.layer9_conv(conv7)
        out = self.fc(conv9).permute(0, 2, 1)
        # outp = self.sigmoid(outp)
        return out


class Discriminator(nn.Module):
    def __init__(self, seq_len, label_dim, cond_data_feature_dim, normalize='LN',
                 **kwargs):
        """
        output_feature_num = num_classes(density map) + 2(x, y)
        """
        super(Discriminator, self).__init__()
        self.label_dim = label_dim
        self.seq_len = seq_len

        self.layer1_conv = double_conv1d_bn(cond_data_feature_dim, 128, kernel_size=7, dim=seq_len, normalize=normalize)
        self.layer2_conv = double_conv1d_bn(128 + label_dim, 32, kernel_size=7, dim=seq_len // 4, normalize=normalize)

        self.layer3_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 8, normalize=normalize)
        self.layer4_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 16, normalize=normalize)

        self.layer5_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 32, normalize=normalize)
        self.layer6_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 64, normalize=normalize)

        self.layer7_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 64, normalize=normalize)
        self.layer8_conv = double_conv1d_bn(32, 32, kernel_size=5, dim=seq_len // 64, normalize=normalize)

        self.fc = nn.Linear(32, 1)

    def forward(self, cond_data, gen_output):
        """
        gen_output: N, L, label_dim
        cond_data: N, L, cond_data_feature_dim
        """
        N, L, _ = cond_data.shape
        # Concatenate label embedding and image to produce input
        # -> N, C, L
        # print('before cnn')
        # print(torch.min(dis_input[0]))
        # print(torch.max(dis_input[0]))
        # print('d_in.shape')
        # print(d_in.shape)
        # -> N, C, fold_len, L // fold_len

        x = self.layer1_conv(cond_data.permute(0, 2, 1))
        x = F.max_pool1d(x, 4)
        x = torch.cat([gen_output.permute(0, 2, 1), x], dim=1)
        # // 4

        x = self.layer2_conv(x)
        x = F.max_pool1d(x, 2)
        # // 8

        x = self.layer3_conv(x)
        x = F.max_pool1d(x, 2)
        # // 16

        x = self.layer4_conv(x)
        x = F.max_pool1d(x, 2)
        # // 32

        x = self.layer5_conv(x)
        x = F.max_pool1d(x, 2)
        # // 64

        x = self.layer6_conv(x)
        x = self.layer7_conv(x)
        x = self.layer8_conv(x)
        # print('after cnn')
        # print(torch.min(validity[0]))
        # print(torch.max(validity[0]))
        validity = F.avg_pool1d(x, x.shape[2]).squeeze(-1)
        # print('after pool')
        # print(torch.min(validity[0]))
        # print(torch.max(validity[0]))

        validity = self.fc(validity)
        # print('after fc')
        # print(torch.min(validity[0]))
        # print(torch.max(validity[0]))

        # N, 1
        return validity
