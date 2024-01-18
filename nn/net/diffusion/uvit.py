import math
from functools import partial, wraps

import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from torch.cuda.amp import autocast

from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange


def exists(val):
    return val is not None


def identity(t):
    return t


def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"


def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d


def cast_tuple(t, l=1):
    return ((t,) * l) if not isinstance(t, tuple) else t


def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))


def l2norm(t):
    return F.normalize(t, dim=-1)


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            factor=2
    ):
        super().__init__()
        self.factor = factor

        dim_out = default(dim_out, dim)
        conv = nn.Conv1d(dim, dim_out * self.factor, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p) l -> b c (l p)', p=factor),
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, l = conv.weight.shape
        conv_weight = torch.empty(o // self.factor, i, l)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=self.factor)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)


def Downsample(
        dim,
        dim_out=None,
        factor=2
):
    return nn.Sequential(
        Rearrange('b c (l p) -> b (c p) l', p=factor),
        nn.Conv1d(dim * factor, default(dim_out, dim), 1)
    )


# https://arxiv.org/abs/1910.07467
class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True, normalize_dim=2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        # channel dim scale, independently for every sample/position
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim=normalize_dim) * scale * (x.shape[normalize_dim] ** 0.5)


# sinusoidal positional embeds

class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # b, 2*(d//2)
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        # b, 2*(d//2)+1
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


# https://arxiv.org/abs/2006.16236
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim=1)
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim, normalize_dim=1)
        )

    def forward(self, x):
        residual = x

        b, c, l = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) l -> b h c l', h=self.heads), qkv)

        # channel dim softmax
        q = q.softmax(dim=-2)
        # spatial dim softmax
        k = k.softmax(dim=-1)

        q = q * self.scale

        # e=d
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c l -> b (h c) l', h=self.heads, l=l)

        return self.to_out(out) + residual


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=8, dropout=0.):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        q, k = map(l2norm, (q, k))

        # channel-wise scale
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.norm = RMSNorm(dim, scale=False)
        dim_hidden = dim * mult

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_hidden * 2),
            Rearrange('b d -> b 1 d')
        )

        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias=False),
            nn.SiLU()
        )

        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias=False)
        )

    def forward(self, x, t):
        x = self.norm(x)
        x = self.proj_in(x)

        scale, shift = self.to_scale_shift(t).chunk(2, dim=-1)
        x = x * (scale + 1) + shift

        return self.proj_out(x)


# vit

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            time_cond_dim,
            depth,
            dim_head=32,
            heads=4,
            ff_mult=4,
            dropout=0.,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=dropout),
                FeedForward(dim=dim, mult=ff_mult, cond_dim=time_cond_dim, dropout=dropout)
            ]))

    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x, t) + x

        return x


# model
class UViT(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            downsample_factor=2,
            channels=3,
            vit_depth=6,
            vit_dropout=0.2,
            attn_dim_head=32,
            attn_heads=4,
            ff_mult=4,
            resnet_block_groups=8,
            learned_sinusoidal_dim=16,
            init_img_transform: callable = None,
            final_img_itransform: callable = None,
            patch_size=1,
            dual_patchnorm=False
    ):
        super().__init__()

        # for initial dwt transform (or whatever transform researcher wants to try here)

        if exists(init_img_transform) and exists(final_img_itransform):
            init_shape = torch.Size(1, 1, 32)
            mock_tensor = torch.randn(init_shape)
            assert final_img_itransform(init_img_transform(mock_tensor)).shape == init_shape

        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        # whether to do initial patching, as alternative to dwt

        self.unpatchify = identity

        input_channels = channels * (patch_size ** 2)
        needs_patch = patch_size > 1

        # if needs_patch:
        #     if not dual_patchnorm:
        #         self.init_conv = nn.Conv2d(channels, init_dim, patch_size, stride=patch_size)
        #     else:
        #         self.init_conv = nn.Sequential(
        #             Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1=patch_size, p2=patch_size),
        #             nn.LayerNorm(input_channels),
        #             nn.Linear(input_channels, init_dim),
        #             nn.LayerNorm(init_dim),
        #             Rearrange('b h w c -> b c h w')
        #         )
        #
        #     self.unpatchify = nn.ConvTranspose2d(input_channels, channels, patch_size, stride=patch_size)

        # determine dimensions

        # [32, 32, 64, 128, 256]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        resnet_block = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # downsample factors

        downsample_factor = cast_tuple(downsample_factor, len(dim_mults))
        assert len(downsample_factor) == len(dim_mults)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                LinearAttention(dim_in),
                Downsample(dim_in, dim_out, factor=factor)
            ]))

        mid_dim = dims[-1]

        self.vit = Transformer(
            dim=mid_dim,
            time_cond_dim=time_dim,
            depth=vit_depth,
            dim_head=attn_dim_head,
            heads=attn_heads,
            ff_mult=ff_mult,
            dropout=vit_dropout
        )

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(reversed(in_out), reversed(downsample_factor))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in, factor=factor),
                resnet_block(dim_in * 2, dim_in, time_emb_dim=time_dim),
                resnet_block(dim_in * 2, dim_in, time_emb_dim=time_dim),
                LinearAttention(dim_in),
            ]))

        default_out_dim = input_channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time):
        # input should be [n, c, l]
        x = self.init_img_transform(x)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = rearrange(x, 'b c l -> b l c')
        # x, ps = pack([x], 'b * c')

        x = self.vit(x, t)

        # x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b l c -> b c l')

        for upsample, block1, block2, attn in self.ups:
            x = upsample(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = self.unpatchify(x)
        return self.final_img_itransform(x)


if __name__ == '__main__':
    uvit = UViT(
        dim=32,
        channels=4,
    )
    # total_numel = 0
    # for param_name, param in uvit.named_parameters():
    #     print(param_name, param.numel())
    #     total_numel += param.numel()
    # print(total_numel)
    inp = torch.ones([2, 4, 32], dtype=torch.float32)
    time = torch.ones([2], dtype=torch.float32)
    print(uvit(inp, time).shape)
