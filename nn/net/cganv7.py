import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv1d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1):
        super(double_conv1d_bn, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=kernel_size,
                               stride=strides, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class deconv1d_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, strides=2):
        super(deconv1d_bn, self).__init__()
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels,
                                        kernel_size=kernel_size,
                                        stride=strides, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class Generator(nn.Module):
    """
    use Unet like structure
    """
    def __init__(self, label_dim, cond_data_feature_dim, noise_dim):
        super(Generator, self).__init__()
        self.layer1_conv = double_conv1d_bn(cond_data_feature_dim + noise_dim, 128, kernel_size=7)
        self.layer2_conv = double_conv1d_bn(128, 32, kernel_size=7)

        self.layer3_conv = double_conv1d_bn(32, 32, kernel_size=5)
        self.layer4_conv = double_conv1d_bn(32, 32, kernel_size=5)

        self.layer5_conv = double_conv1d_bn(32, 32, kernel_size=5)

        self.layer6_conv = double_conv1d_bn(64, 32, kernel_size=5)
        self.layer7_conv = double_conv1d_bn(64, 32, kernel_size=5)
        # self.layer8_conv = double_conv1d_bn(64, 32, kernel_size=3)
        # self.layer9_conv = double_conv1d_bn(16, 8)

        self.deconv1 = deconv1d_bn(32, 32, kernel_size=4, strides=4)
        self.deconv2 = deconv1d_bn(32, 32, kernel_size=4, strides=4)
        # self.deconv3 = deconv1d_bn(32, 32, kernel_size=4, strides=4)
        # self.deconv4 = deconv1d_bn(16, 8)

        self.layer9_conv = double_conv1d_bn(32, 32, kernel_size=5)
        self.fc = nn.Linear(32, label_dim)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.layer1_conv(x)
        pool1 = F.max_pool2d(conv1, 2)

        conv2 = self.layer2_conv(pool1)
        pool2 = F.max_pool2d(conv2, 2)  # length of pool2 = length of output = 1/4 * length of input = 384*4 / 4 = 384

        conv3 = self.layer3_conv(pool2)
        pool3 = F.max_pool2d(conv3, 4)

        conv4 = self.layer4_conv(pool3)
        pool4 = F.max_pool2d(conv4, 4)

        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1, conv4], dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2, conv3], dim=1)
        conv7 = self.layer7_conv(concat2)

        conv9 = self.layer9_conv(conv7)
        outp = self.layer10_conv(conv9)
        # outp = self.sigmoid(outp)
        return outp


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

        dims = [seq_len // self.fold_len, self.fold_len]

        # self.ext = FeatureExtractor(self.ext_in_channels, self.ext_out_channels)

        self.layer1_conv = double_conv1d_bn(cond_data_feature_dim + label_dim, 128, kernel_size=7)
        self.layer2_conv = double_conv1d_bn(128, 32, kernel_size=7)

        self.layer3_conv = double_conv1d_bn(32, 32, kernel_size=5)
        self.layer4_conv = double_conv1d_bn(32, 32, kernel_size=5)

        self.fc = nn.Linear(64, 1)

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
        x = torch.cat([gen_output, cond_data], dim=2).permute(0, 2, 1).reshape([N, -1, L // self.fold_len, self.fold_len])
        # print('before cnn')
        # print(torch.min(dis_input[0]))
        # print(torch.max(dis_input[0]))
        # print('d_in.shape')
        # print(d_in.shape)
        # -> N, C, fold_len, L // fold_len

        x = self.layer1_conv(x)
        x = F.max_pool1d(x, 4)

        x = self.layer2_conv(x)
        x = F.max_pool1d(x, 4)

        x = self.layer3_conv(x)
        x = F.max_pool1d(x, 4)

        x = self.layer4_conv(x)
        # x = F.max_pool1d(x, 4)

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
