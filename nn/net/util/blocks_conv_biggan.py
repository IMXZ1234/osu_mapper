from nn.net.util.transformer_modules import PosEncoding
from torch import nn
from torch.nn import functional as F
import torch
from torchvision.models import resnet


# Projection of x onto y
def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        # svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs


# Spectral normalization base class
class SN(object):
    def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values;
    # note that these buffers are just for logging and are not used in training.
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
            # Update the svs
        if self.training:
            with torch.no_grad():  # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]


# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    
# Generator blocks
# Note that this class assumes the kernel size and padding (and any other
# settings) have been selected in the main generator module and passed in
# through the which_conv arg. Similar rules apply with which_bn (the input
# size [which is actually the number of channels of the conditional info] must 
# be preselected)
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=bn, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                 preactivation=False, activation=None, downsample=None, ):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


class ConvDownSampleBlock1DNoBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len,
                 stride=2, kernel_size=5,
                 norm='LN', dropout=0, residual=True):
        super().__init__()
        self.stride = stride
        self.convnext = ConvBlock1DNoBottleneck(
            in_channels, out_channels, seq_len * stride,
            1, kernel_size,
            norm, dropout, residual
        )

    def forward(self, x):
        """x: n, c, l"""
        x = self.convnext(x)
        N, C, L = x.shape
        x = torch.mean(x.reshape(N, C, L // self.stride, self.stride), dim=-1)
        return x


class CNNExtractor(nn.Module):
    def __init__(self, in_channels, seq_len,
                 stride_list=(2, 2, 2, 2), out_channels_list=(128, 128, 128, 128),
                 kernel_size_list=(5, 5, 5, 5),
                 norm='LN', input_norm=False, first_layer_residual=True):
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
        residual_list = [first_layer_residual] + [True] * (len(stride_list) - 1)
        for stride, out_channels, kernel_size, residual in zip(
                stride_list, out_channels_list, kernel_size_list, residual_list):
            self.net.append(
                ConvBlock1DNoBottleneck(
                    current_in_channels,
                    out_channels,
                    current_seq_len,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm,
                    residual=residual,
                )
            )
            current_in_channels = out_channels
            current_seq_len = current_seq_len // stride
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class CNNExtractorDownsample(nn.Module):
    def __init__(self, in_channels, seq_len,
                 stride_list=(2, 2, 2, 2), out_channels_list=(128, 128, 128, 128),
                 kernel_size_list=(5, 5, 5, 5),
                 norm='LN', input_norm=False, first_layer_residual=True):
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
        residual_list = [first_layer_residual] + [True] * (len(stride_list) - 1)
        for stride, out_channels, kernel_size, residual in zip(
                stride_list, out_channels_list, kernel_size_list, residual_list):
            self.net.append(
                ConvBlock1DNoBottleneck(
                    current_in_channels,
                    out_channels,
                    current_seq_len,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm,
                    residual=residual,
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
                 first_layer_residual=True,
                 dropout=0):
        super().__init__()
        self.net = []
        current_in_channels = in_channels
        current_seq_len = seq_len
        residual_list = [first_layer_residual] + [True] * (len(stride_list) - 1)
        for stride, out_channels, kernel_size, residual in zip(
                stride_list, out_channels_list, kernel_size_list, residual_list):
            self.net.append(
                ConvUpSampleBlock1DNoBottleneck(
                    current_in_channels,
                    out_channels,
                    current_seq_len,
                    stride=stride,
                    kernel_size=kernel_size,
                    norm=norm,
                    dropout=dropout,
                    residual=residual,
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