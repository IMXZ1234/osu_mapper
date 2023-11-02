import torch
from torch import nn
import numpy as np


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec, batch_first=False):
        """

        """
        super(PosEncoding, self).__init__()
        self.batch_first = batch_first
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
             for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])

        self.pos_enc = nn.Embedding(max_seq_len, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)

    def forward(self, x):
        """x: n, l, c(batch_first=True) or l, n, c(batch_first=False)"""
        # -> n, l, c
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        input_pos = torch.arange(x.shape[1], device=x.device)

        enc = self.pos_enc(input_pos).float()
        x = enc + x
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        return x
