import random

import torch
from torch import nn, autograd
from torch.nn import functional as F
from nn.net.util.blocks_conv import CNNExtractor, DecoderUpsample


def gen_pos_seq(batch_size, seq_len, period=8, device='cpu'):
    """
    -> batch_size, seq_len
    """
    return (torch.arange(seq_len, device=device, dtype=torch.long) % period).repeat(batch_size, 1)


class VAE(nn.Module):
    """compression of hit_type and coord into latent"""
    def __init__(self, n_tokens, discrete_in_channels, continuous_in_channels,
                 enc_stride_list, enc_out_channels_list,
                 dec_stride_list=None, dec_out_channels_list=None,
                 hidden_dim=None):
        super(VAE, self).__init__()
        self.discrete_in_channels = discrete_in_channels
        self.n_tokens = n_tokens
        self.continuous_in_channels = continuous_in_channels
        self.in_channels = discrete_in_channels + continuous_in_channels

        self.embedding = nn.Embedding(n_tokens, discrete_in_channels)

        self.encoder = CNNExtractor(
            discrete_in_channels + continuous_in_channels,
            enc_stride_list,
            enc_out_channels_list,
            kernel_size_list=[3] * len(enc_stride_list),
            norm='BN',
            input_norm=True,
        )
        if dec_stride_list is None:
            dec_stride_list = enc_stride_list
        if dec_out_channels_list is None:
            dec_out_channels_list = list(reversed(enc_out_channels_list[:-1])) + [n_tokens + continuous_in_channels]

        self.proj_dim = enc_out_channels_list[-1]

        self.decoder = DecoderUpsample(
            self.proj_dim,
            dec_stride_list,
            dec_out_channels_list,
            kernel_size_list=[3] * len(dec_stride_list),
            norm='BN',
        )
        self.hidden_dim = self.proj_dim if hidden_dim is None else hidden_dim

        self.fc_mu = nn.Conv1d(self.proj_dim, self.hidden_dim, kernel_size=1)
        self.fc_log_std = nn.Conv1d(self.proj_dim, self.hidden_dim, kernel_size=1)

        self.fc_back_proj = nn.Conv1d(self.hidden_dim, self.proj_dim, kernel_size=1)

    def decode(self, z):
        x = self.fc_back_proj(z)
        recon = self.decoder(x)
        out_discrete, out_continuous = recon[:, :self.n_tokens], recon[:, self.n_tokens:]
        return out_discrete, out_continuous

    def encode(self, x):
        """
        inp_discrete: [N, L]
        inp_continuous: [N, C, L]
        """
        inp_discrete, inp_continuous = x
        # N, L, C -> N, C, L
        word_embed = self.embedding(inp_discrete).permute(0, 2, 1)
        inp = torch.cat([word_embed, inp_continuous], dim=1)
        hidden = self.encoder(inp)
        mu = self.fc_mu(hidden)
        log_std = self.fc_log_std(hidden)
        return mu, log_std

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # sample from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, x):
        """
        inp_discrete: [N, L]
        inp_continuous: [N, C, L]

        returns

        out_discrete: [N, Cd, L]  probability distribution
        out_continuous: [N, C, L]
        """
        mu, log_std = self.encode(x)

        z = self.reparametrize(mu, log_std)

        recon = self.decode(z)

        return recon, mu, log_std

    def loss_function(self, recon, mu, log_std, label):
        inp_discrete, inp_continuous = label
        out_discrete, out_continuous = recon

        continuous_loss = F.mse_loss(inp_continuous, out_continuous, reduction="mean")
        discrete_loss = F.cross_entropy(out_discrete, inp_discrete)
        # loss = continuous_loss + discrete_loss
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        # print('loss %.4f' % loss.item())
        return (discrete_loss, continuous_loss), kl_loss

    def sample(self, batch_size, seq_len):

        z = torch.randn([batch_size, self.hidden_dim, seq_len], device=next(self.parameters()).device)

        return self.decode(z)
