import torch
from torch import nn


class Word2VecEmbed(nn.Module):
    def __init__(self, vocab_size, feature_dim, train=True):
        super().__init__()
        # self.embedding = nn.Embedding()
        embedding_center = torch.randn(vocab_size, feature_dim)
        embedding_context = torch.randn(vocab_size, feature_dim)
        if train:
            self.embedding_center = nn.Parameter(embedding_center)
            self.embedding_context = nn.Parameter(embedding_context)
        else:
            self.register_buffer('embedding_center', embedding_center)
            self.register_buffer('embedding_context', embedding_context)

    def forward(self, label_idx):
        return self.embedding_center[label_idx]

    def forward_context(self, label_idx):
        return self.embedding_context[label_idx]
