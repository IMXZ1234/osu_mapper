import torch
from torch import nn


class Word2VecEmbed(nn.Module):
    def __init__(self, vocab_size, feature_dim, train=True):
        super().__init__()
        # self.embedding = nn.Embedding()
        embedding = torch.randn(vocab_size, feature_dim)
        if train:
            self.embedding = nn.Parameter(embedding)
        else:
            self.register_buffer('embedding', embedding)

    def forward(self, label_idx):
        return self.embedding[label_idx]
