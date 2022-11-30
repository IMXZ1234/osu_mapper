"""
added pos encode
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, cond_data_feature_dim, vocab_size, seq_len, num_layers=1):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(seq_len, embedding_dim)
        # action embedding + pos embedding + cond_data
        self.gru = nn.GRU(embedding_dim + embedding_dim + cond_data_feature_dim, hidden_dim, num_layers=num_layers)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size=1, device='cpu'):
        return autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))

    def forward(self, cond_data, inp, hidden, pos):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        cond_data: N, feature_dim=514
        inp: N, (label)
        hidden: N, hidden_dim
        pos: N, (label)
        """
        batch_size, cond_data_feature_dim = cond_data.shape
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, batch_size, self.embedding_dim)               # 1 x batch_size x embedding_dim
        pos_emb = self.pos_embeddings(pos % self.seq_len)                              # batch_size x embedding_dim
        pos_emb = pos_emb.view(1, batch_size, self.embedding_dim)               # 1 x batch_size x embedding_dim
        cond_data = cond_data.view(1, batch_size, cond_data_feature_dim)
        out, hidden = self.gru(torch.cat([emb, pos_emb, cond_data], dim=2), hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, cond_data, h=None, start_letter=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        cond_data: num_samples, seq_len, feature_dim

        Outputs: samples, hidden
            - samples: num_samples x seq_len (a sampled sequence in each row)
        """
        num_samples, seq_len, feature_dim = cond_data.shape
        samples = torch.zeros([num_samples, seq_len], dtype=torch.long, device=cond_data.device)

        if h is None:
            h = self.init_hidden(num_samples, cond_data.device)
        inp = autograd.Variable(torch.tensor([start_letter]*num_samples, dtype=torch.long, device=cond_data.device))

        for i in range(seq_len):
            pos = torch.ones([num_samples], dtype=torch.long, device=cond_data.device) * i
            out, h = self.forward(cond_data[:, i], inp, h, pos)               # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples, h

    def batchNLLLoss(self, cond_data, inp, target, h=None):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - cond_data: batch_size x seq_len x feature_dim
            - inp: batch_size x seq_len
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)           # seq_len x batch_size
        target = target.permute(1, 0)     # seq_len x batch_size
        cond_data = cond_data.permute(1, 0, 2)
        if h is None:
            h = self.init_hidden(batch_size, device=cond_data.device)

        loss = 0
        for i in range(seq_len):
            pos = torch.ones([batch_size], dtype=torch.long, device=cond_data.device) * i
            out, h = self.forward(cond_data[i], inp[i], h, pos)
            loss += loss_fn(out, target[i])

        return loss, h

    def batchPGLoss(self, cond_data, inp, target, reward, h=None):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: inp, target
            - cond_data: batch_size x seq_len x feature_dim
            - inp: batch_size x seq_len
            - target: batch_size x seq_len
            - reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding
                      sentence)

            inp should be target with <s> (start letter) prepended
        """

        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)          # seq_len x batch_size
        target = target.permute(1, 0)    # seq_len x batch_size
        cond_data = cond_data.permute(1, 0, 2)
        if h is None:
            h = self.init_hidden(batch_size, device=cond_data.device)

        loss = 0
        for i in range(seq_len):
            pos = torch.ones([batch_size], dtype=torch.long, device=cond_data.device) * i
            out, h = self.forward(cond_data[i], inp[i], h, pos)
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size, h


class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, cond_data_feature_dim, vocab_size, seq_len, dropout=0.2, num_layers=2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(seq_len, embedding_dim)
        self.gru = nn.GRU(embedding_dim + embedding_dim + cond_data_feature_dim, hidden_dim,
                          num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*self.num_layers*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size, device='cpu'):
        return autograd.Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim, device=device))

    def forward(self, cond_data, inp, h):
        batch_size, total_len = inp.shape
        cond_data = cond_data.permute(1, 0, 2)
        # 1 x total_len -> batch_size, total_len
        pos = (torch.arange(total_len, device=cond_data.device, dtype=torch.long) % self.seq_len).reshape([1, total_len]).expand_as(inp)
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(inp)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        pos_emb = self.pos_embeddings(pos)                               # batch_size x seq_len x embedding_dim
        pos_emb = pos_emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, h = self.gru(torch.cat([emb, pos_emb, cond_data], dim=2), h)                          # 4 x batch_size x hidden_dim
        h = h.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(h.view(-1, 2 * self.num_layers * self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out, h

    def batchClassify(self, cond_data, inp, h=None):
        """
        Classifies a batch of sequences.

        Inputs: cond_data, inp
            - cond_data: batch_size x seq_len x feature_dim
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        if h is None:
            h = self.init_hidden(inp.size()[0], device=cond_data.device)
        out, h = self.forward(cond_data, inp, h)
        return out.view(-1), h

    # def batchBCELoss(self, cond_data, inp, target):
    #     """
    #     Returns Binary Cross Entropy Loss for discriminator.
    #
    #      Inputs: cond_data, inp, target
    #         - cond_data: batch_size x seq_len x feature_dim
    #         - inp: batch_size x seq_len
    #         - target: batch_size (binary 1/0)
    #     """
    #
    #     loss_fn = nn.BCELoss()
    #     h = self.init_hidden(inp.size()[0], device=cond_data.device)
    #     out = self.forward(cond_data, inp, h)
    #     return loss_fn(out, target)

