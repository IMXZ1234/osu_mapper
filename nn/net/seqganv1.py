import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, cond_data_feature_dim, vocab_size):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim + cond_data_feature_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size=1, device='cpu'):
        return autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim, device=device))

    def forward(self, cond_data, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        cond_data: N, feature_dim=514
        inp: N, (label)
        hidden: N, hidden_dim
        """
        batch_size, cond_data_feature_dim = cond_data.shape
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, batch_size, self.embedding_dim)               # 1 x batch_size x embedding_dim
        cond_data = cond_data.view(1, batch_size, cond_data_feature_dim)
        out, hidden = self.gru(torch.cat([emb, cond_data], dim=2), hidden)                     # 1 x batch_size x hidden_dim (out)
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
        samples = torch.zeros(num_samples, seq_len).type(torch.LongTensor)

        if h is None:
            h = self.init_hidden(num_samples, cond_data.device)
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples), device=cond_data.device)

        for i in range(self.max_seq_len):
            out, h = self.forward(cond_data[:, i], inp, h)               # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples

    def batchNLLLoss(self, cond_data, inp, target):
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
        h = self.init_hidden(batch_size)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(cond_data[i], inp[i], h)
            loss += loss_fn(out, target[i])

        return loss     # per batch

    def batchPGLoss(self, cond_data, inp, target, reward):
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
        h = self.init_hidden(batch_size, device=cond_data.device)

        loss = 0
        for i in range(seq_len):
            out, h = self.forward(cond_data[i], inp[i], h)
            # TODO: should h be detached from graph (.detach())?
            for j in range(batch_size):
                loss += -out[j][target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return loss/batch_size


class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, cond_data_feature_dim, vocab_size, dropout=0.2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim + cond_data_feature_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size, device='cpu'):
        return autograd.Variable(torch.zeros(2*2*1, batch_size, self.hidden_dim, device=device))

    def forward(self, cond_data, inp, hidden):
        cond_data = cond_data.permute(1, 0, 2)
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(inp)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(torch.cat([emb, cond_data], dim=2), hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out

    def batchClassify(self, cond_data, inp):
        """
        Classifies a batch of sequences.

        Inputs: cond_data, inp
            - cond_data: batch_size x seq_len x feature_dim
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """

        h = self.init_hidden(inp.size()[0], device=cond_data.device)
        out = self.forward(cond_data, inp, h)
        return out.view(-1)

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

