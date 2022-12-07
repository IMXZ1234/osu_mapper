import random

import torch
from torch import nn, autograd
from torch.nn import functional as F


def gen_pos_seq(batch_size, seq_len, period=8, device='cpu'):
    """
    -> batch_size, seq_len
    """
    return (torch.arange(seq_len, device=device, dtype=torch.long) % period).repeat(batch_size, 1)


class VAE(nn.Module):
    """Implementation of VAE(Variational Auto-Encoder)"""
    def __init__(self, in_channels, cond_data_feature_dim, hidden_dim,
                 pos_emb_period=8, pos_emb_channels=4,
                 enc_layers=2, dec_layers=1):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.enc_layers, self.dec_layers = enc_layers, dec_layers
        self.pos_emb_period = pos_emb_period

        # out_dim = seq_len * in_channels
        enc_h_dim = hidden_dim * 2 * self.enc_layers
        dec_h_dim = hidden_dim * 1 * self.dec_layers
        h_dim = 512

        self.pos_embedding = nn.Embedding(pos_emb_period, pos_emb_channels)

        # encoder
        self.gru_enc = nn.GRU(cond_data_feature_dim + pos_emb_channels + in_channels, hidden_dim, bidirectional=True, num_layers=enc_layers)
        self.z_dim = h_dim
        self.shrink_h = nn.Linear(enc_h_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, self.z_dim)
        self.fc_log_std = nn.Linear(h_dim, self.z_dim)

        # decoder
        self.fc_z = nn.Linear(self.z_dim, dec_h_dim)
        self.gru_dec = nn.GRU(cond_data_feature_dim + pos_emb_channels + in_channels, hidden_dim, bidirectional=False, num_layers=dec_layers)
        self.fc_dec = nn.Linear(self.hidden_dim, in_channels)

    def init_enc_hidden(self, batch_size=1, device='cpu'):
        return autograd.Variable(torch.zeros(2*self.enc_layers, batch_size, self.hidden_dim, device=device))

    def encode(self, cond_data, label, pos_emb):
        batch_size, seq_len, _ = cond_data.shape
        gru_input = torch.cat([cond_data, label, pos_emb], dim=2).permute(1, 0, 2)
        h = self.init_enc_hidden(batch_size, cond_data.device)
        _, h = self.gru_enc(gru_input, h)

        h = h.permute(1, 0, 2).reshape([batch_size, -1])
        h = F.relu(self.shrink_h(h))

        mu = self.fc_mu(h)
        log_std = self.fc_log_std(h)

        return mu, log_std

    def decode(self, z, cond_data, pos_emb, label=None, teacher_forcing_ratio=0.5):
        batch_size, seq_len, _ = cond_data.shape
        h = self.fc_z(z).reshape([batch_size, self.dec_layers, -1]).permute(1, 0, 2)
        all_out = []

        out = torch.ones([batch_size, self.in_channels], dtype=torch.float, device=cond_data.device) / 2
        for i in range(seq_len):
            gru_input = torch.cat([cond_data[:, i], out, pos_emb[:, i]], dim=1).unsqueeze(dim=0)
            out, h = self.gru_dec(gru_input, h)
            out = out.squeeze(dim=0)
            out = self.fc_dec(out)
            all_out.append(out)
            if label is not None and random.random() < teacher_forcing_ratio:
                out = label[:, i]
        recon = torch.stack(all_out, dim=1)
        return recon, h

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, cond_data, label, teacher_forcing_ratio=0.5):
        """
        label: [batch_size, seq_len, 4(circle_density, slider_density, x, y)]
        cond_data: [batch_size, seq_len, cond_data_feature_dim]
        """
        batch_size, seq_len, _ = cond_data.shape
        pos_idx = gen_pos_seq(batch_size, seq_len, self.pos_emb_period, cond_data.device)
        # N, L -> N, L, pos_emb_channels
        pos_emb = self.pos_embedding(pos_idx)

        mu, log_std = self.encode(cond_data, label, pos_emb)

        z = self.reparametrize(mu, log_std)
        # print(torch.isnan(z).any())

        recon, h = self.decode(z, cond_data, pos_emb, label, teacher_forcing_ratio)

        return recon, mu, log_std

    def loss_function(self, recon, mu, log_std, label):
        # print('recon[0]')
        # print(recon[0])
        # x = inp_to_one_hot(x, self.num_class)
        # print('x[0]')
        # print(x[0])
        circle_loss = F.mse_loss(recon[:, :, 0], label[:, :, 0], reduction="mean")
        slider_loss = F.mse_loss(recon[:, :, 1], label[:, :, 1], reduction="mean")
        pos_loss = F.mse_loss(recon[:, :, 2:], label[:, :, 2:], reduction="mean")
        loss = circle_loss * 25 + slider_loss + pos_loss * 25
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        print('loss %.4f' % loss.item())
        return loss, kl_loss

    def sample(self, cond_data):
        batch_size, seq_len, _ = cond_data.shape
        pos_idx = gen_pos_seq(batch_size, seq_len, self.pos_emb_period, cond_data.device)
        # N, L -> N, L, pos_emb_channels
        pos_emb = self.pos_embedding(pos_idx)

        z = torch.randn([batch_size, self.z_dim], device=cond_data.device)

        recon, h = self.decode(z, cond_data, pos_emb, None, teacher_forcing_ratio=0)

        return recon
