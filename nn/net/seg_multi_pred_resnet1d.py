"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from nn.net.util import crop1d


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class SegMultiPredResNet1D(nn.Module):

    def __init__(self, num_block=(1, 1, 1, 1), stride=(4, 4, 4, 4),
                 num_classes=2, seg_len=16384//8, keep_seg=8*8, block='BasicBlock'):
        super().__init__()

        if block == 'BasicBlock':
            block = BasicBlock
        else:
            block = 'BottleNeck'

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], stride[0])
        self.conv3_x = self._make_layer(block, 128, num_block[1], stride[1])
        self.conv4_x = self._make_layer(block, 256, num_block[2], stride[2])
        self.conv5_x = self._make_layer(block, 512, num_block[3], stride[3])
        total_shrink = torch.prod(torch.tensor(stride)).item()
        final_seg_len = seg_len // total_shrink
        self.crop = crop1d.Crop1D(final_seg_len * keep_seg)
        self.pool = nn.MaxPool1d(kernel_size=final_seg_len, stride=final_seg_len)
        self.fc1 = nn.Linear(512 * block.expansion + 1, 64)
        self.fc_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # N, feature_num
        x, speed_stars = x
        # print('x.shape')
        # print(x.shape)
        # print('speed_stars')
        # print(speed_stars)

        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        # print('before crop x.shape')
        # print(x.shape)
        x = self.crop(x)
        # print('after crop x.shape')
        # print(x.shape)
        x = self.pool(x)
        # print('pool x.shape')
        # print(x.shape)
        N, C, L = x.shape
        speed_stars_feature = torch.tensor(speed_stars,
                                           device=x.device,
                                           dtype=torch.float)
        speed_stars_feature = speed_stars_feature.reshape([N, 1, 1]).expand([N, L, 1]).reshape([N*L, 1])
        x = x.permute([0, 2, 1]).reshape([N*L, C])
        # cat along feature dim
        x = torch.cat([x, speed_stars_feature], dim=1)
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.fc2(x)
        x = x.reshape([N, L, -1])
        # print('out x')
        # print(x.shape)

        return x
