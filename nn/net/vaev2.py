import torch
from torch import nn, autograd
from torch.nn import functional as F


def inp_to_one_hot(inp, num_class):
    """
    change type_label in inp to one_hot
    """
    # batch_size = inp.shape[0]
    # print('inp.shape')
    # print(inp.shape)
    type_label, ho_pos = inp[:, :, 0], inp[:, :, 1:]
    type_label = F.one_hot(type_label.long(), num_classes=num_class).float()
    # print('type_label.shape')
    # print(type_label.shape)
    # print('ho_pos.shape')
    # print(ho_pos.shape)
    return torch.cat([type_label, ho_pos], dim=2)


class VAE(nn.Module):
    """Implementation of VAE(Variational Auto-Encoder)"""
    def __init__(self, num_class, seq_len, cond_data_feature_dim, hidden_dim, num_layers=1):
        super(VAE, self).__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(cond_data_feature_dim, hidden_dim, bidirectional=True, num_layers=num_layers)

        out_dim = seq_len * (num_class + 2)
        out_h_dim = hidden_dim * 2*self.num_layers
        h_dim = 512
        self.z_dim = h_dim
        self.shrink_h = nn.Linear(out_h_dim, h_dim)
        total_feature_dim = out_dim + h_dim
        self.fc1 = nn.Linear(total_feature_dim, h_dim)
        self.fc1_1 = nn.Linear(h_dim, h_dim)
        self.fc2_mu = nn.Linear(h_dim, self.z_dim)
        self.fc2_log_std = nn.Linear(h_dim, self.z_dim)
        self.fc3 = nn.Linear(self.z_dim, h_dim)
        # self.fc4 = nn.Linear(h_dim + h_dim, h_dim + h_dim)
        self.fc5 = nn.Linear(h_dim, out_dim)

    def init_hidden(self, batch_size=1, device='cpu'):
        return autograd.Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim, device=device))

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1_1(h1))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    # def decode(self, z, h):
    #     h3 = F.relu(self.fc3(z))
    #     h3 = torch.cat([h3, h], dim=1)
    #     # h4 = F.relu(self.fc4(h3))
    #     recon = torch.sigmoid(self.fc5(h3))  # use sigmoid because the input image's pixel is between 0-1
    #     return recon

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # h4 = F.relu(self.fc4(h3))
        recon = torch.sigmoid(self.fc5(h3))  # use sigmoid because the input image's pixel is between 0-1
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # simple from standard normal distribution
        z = mu + eps * std
        return z

    def forward(self, cond_data, x):
        """
        x: type_label [batch_size, seq_len, 3]
        cond_data: [batch_size, seq_len, cond_data_feature_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = inp_to_one_hot(x, self.num_class).reshape([batch_size, -1])
        # print('x.shape')
        # print(x.shape)
        # print('cond_data')
        # print(cond_data[0])
        # print('cond_data.shape')
        # print(cond_data.shape)

        h = self.init_hidden(batch_size, cond_data.device)
        # print('h.shape')
        # print(h.shape)
        _, h = self.gru(cond_data.permute(1, 0, 2), h)
        h = h.permute(1, 0, 2).reshape([batch_size, -1])
        # print('h.shape')
        # print(h.shape)
        h = self.shrink_h(h)
        # print(torch.isnan(h).any())

        encode_input = torch.cat([h, x], dim=1)
        mu, log_std = self.encode(encode_input)

        z = self.reparametrize(mu, log_std)
        # print(torch.isnan(z).any())

        recon = self.decode(z)
        recon = recon.reshape([batch_size, seq_len, -1])
        # print(torch.isnan(recon).any())

        return recon, mu, log_std

    def loss_function(self, recon, mu, log_std, x):
        # print('recon[0]')
        # print(recon[0])
        # x = inp_to_one_hot(x, self.num_class)
        # print('x[0]')
        # print(x[0])
        type_label, ho_pos = x[:, :, 0].long().reshape(-1), x[:, :, 1:]
        pred_type_label_prob, pred_ho_pos = recon[:, :, :self.num_class].reshape([-1, self.num_class]), recon[:, :, self.num_class:]
        type_label_loss = F.cross_entropy(pred_type_label_prob, type_label, reduction="mean")
        ho_pos_loss = F.mse_loss(pred_ho_pos, ho_pos, reduction="mean")
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        print('ho_pos_loss %.4f, type_label_loss %.4f, kl_loss %.4f' % (ho_pos_loss.item(), type_label_loss.item(), kl_loss.item()))
        return (type_label_loss + ho_pos_loss), kl_loss

    def sample(self, cond_data):
        batch_size, seq_len, _ = cond_data.shape
        z = torch.randn([batch_size, self.z_dim], device=cond_data.device)
        h = self.init_hidden(batch_size, cond_data.device)
        _, h = self.gru(cond_data.permute(1, 0, 2), h)
        h = h.permute(1, 0, 2).reshape([batch_size, -1])
        # print('h.shape')
        # print(h.shape)
        h = self.shrink_h(h)
        recon = self.decode(z)
        recon = recon.reshape([batch_size, seq_len, -1])
        return recon
