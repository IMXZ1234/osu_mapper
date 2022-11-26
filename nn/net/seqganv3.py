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
        self.ho_pos_embeddings = nn.Linear(2, embedding_dim)
        # action embedding + pos embedding + cond_data
        self.gru = nn.GRU(embedding_dim*3 + cond_data_feature_dim, hidden_dim, num_layers=num_layers)
        self.gru2out_label = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        self.gru2out_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def init_hidden(self, batch_size=1, device='cpu'):
        return autograd.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))

    def forward(self, cond_data, type_label, ho_pos, hidden, pos):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        cond_data: N, feature_dim=514
        inp: N, (label)
        ho_pos: N, 2
        hidden: N, hidden_dim
        pos: N, (label)
        """
        batch_size, cond_data_feature_dim = cond_data.shape
        # input dim                                             # batch_size
        emb = self.embeddings(type_label)                              # batch_size x embedding_dim
        emb = emb.view(1, batch_size, self.embedding_dim)               # 1 x batch_size x embedding_dim

        pos_emb = self.pos_embeddings(pos % self.seq_len)                              # batch_size x embedding_dim
        pos_emb = pos_emb.view(1, batch_size, self.embedding_dim)               # 1 x batch_size x embedding_dim

        ho_pos_emb = self.ho_pos_embeddings(ho_pos)
        ho_pos_emb = ho_pos_emb.view(1, batch_size, self.embedding_dim)

        cond_data = cond_data.view(1, batch_size, cond_data_feature_dim)
        out, hidden = self.gru(torch.cat([emb, pos_emb, ho_pos_emb, cond_data], dim=2), hidden)                     # 1 x batch_size x hidden_dim (out)
        out = out.view(-1, self.hidden_dim)
        out_label = self.gru2out_label(out)       # batch_size x vocab_size
        type_label_out = F.log_softmax(out_label, dim=1)
        out_pos = self.gru2out_pos(out)       # batch_size x 2
        ho_pos_out = torch.sigmoid(out_pos)
        return (type_label_out, ho_pos_out), hidden

    def sample(self, cond_data, h=None, start_letter=0, start_ho_pos=(0.5, 0.5)):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        cond_data: num_samples, seq_len, feature_dim

        Outputs: samples, hidden
            - samples: num_samples x seq_len (a sampled sequence in each row)
        """
        num_samples, seq_len, feature_dim = cond_data.shape
        type_label_samples = torch.zeros([num_samples, seq_len], dtype=torch.long, device=cond_data.device)
        ho_pos_samples = []

        if h is None:
            h = self.init_hidden(num_samples, cond_data.device)
        type_label = autograd.Variable(torch.tensor([start_letter]*num_samples, dtype=torch.long, device=cond_data.device))
        ho_pos = autograd.Variable(torch.tensor([start_ho_pos]*num_samples, dtype=torch.float, device=cond_data.device))

        for i in range(seq_len):
            pos = torch.ones([num_samples], dtype=torch.long, device=cond_data.device) * i
            # num_samples x 2, num_samples x 2
            (type_label_out, ho_pos_out), hidden = self.forward(cond_data[:, i], type_label, ho_pos, h, pos)               # out: num_samples x vocab_size
            type_label_out = torch.multinomial(torch.exp(type_label_out), 1)  # num_samples x 1 (sampling from each row)
            type_label_samples[:, i] = type_label_out.squeeze(dim=1).data
            ho_pos_samples.append(ho_pos_out)

            type_label = type_label_out.view(-1)
            ho_pos = ho_pos_out
        # -> num_samples x seq_len x 2
        ho_pos_samples = torch.stack(ho_pos_samples, dim=1)
        return (type_label_samples, ho_pos_samples), h

    def batchNLLLoss(self, cond_data, inp, target, h=None):
        """
        Returns the NLL Loss for predicting target sequence.

        Inputs: inp, target
            - cond_data: batch_size x seq_len x feature_dim
            - inp: batch_size x seq_len x 3
            - target: batch_size x seq_len

            inp should be target with <s> (start letter) prepended
        """
        loss_fn_type = nn.NLLLoss()
        loss_fn_pos = nn.MSELoss()
        coeff = 1

        inp = inp.permute(1, 0, 2)
        type_label, ho_pos = inp[:, :, 0].long(), inp[:, :, 1:]
        seq_len, batch_size = type_label.size()

        target = target.permute(1, 0, 2)
        type_label_target, ho_pos_target = target[:, :, 0].long(), target[:, :, 1:]

        cond_data = cond_data.permute(1, 0, 2)
        if h is None:
            h = self.init_hidden(batch_size, device=cond_data.device)

        type_loss, pos_loss = 0, 0
        for i in range(seq_len):
            pos = torch.ones([batch_size], dtype=torch.long, device=cond_data.device) * i
            (type_label_out, ho_pos_out), h = self.forward(cond_data[i], type_label[i], ho_pos[i], h, pos)
            type_loss += loss_fn_type(type_label_out, type_label_target[i])
            # do not predict pos during pretrain as pos greatly differs in beatmaps even for the same audio
            # pos_loss += loss_fn_pos(ho_pos_out, ho_pos_target[i]) * coeff

        # print('type_loss')
        # print(type_loss)
        # print('pos_loss')
        # print(pos_loss)
        return type_loss + pos_loss, h

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
        inp = inp.permute(1, 0, 2)
        type_label, ho_pos = inp[:, :, 0].long(), inp[:, :, 1:]
        seq_len, batch_size = type_label.size()

        target = target.permute(1, 0, 2)
        type_label_target, ho_pos_target = target[:, :, 0].long(), target[:, :, 1:]

        cond_data = cond_data.permute(1, 0, 2)
        if h is None:
            h = self.init_hidden(batch_size, device=cond_data.device)

        pg_loss = 0
        for i in range(seq_len):
            pos = torch.ones([batch_size], dtype=torch.long, device=cond_data.device) * i
            (type_label_out, ho_pos_out), h = self.forward(cond_data[i], type_label[i], ho_pos[i], h, pos)
            for j in range(batch_size):
                pg_loss += -type_label_out[j][type_label_target.data[i][j]]*reward[j]     # log(P(y_t|Y_1:Y_{t-1})) * Q

        return pg_loss/batch_size, h


class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, cond_data_feature_dim, vocab_size, seq_len, dropout=0.2, num_layers=2):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embeddings = nn.Embedding(seq_len, embedding_dim)
        self.ho_pos_embeddings = nn.Linear(2, embedding_dim)
        self.gru = nn.GRU(embedding_dim*3 + cond_data_feature_dim, hidden_dim,
                          num_layers=num_layers, bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2*self.num_layers*hidden_dim, hidden_dim)
        self.dropout_linear = nn.Dropout(p=dropout)
        self.hidden2out = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size, device='cpu'):
        return autograd.Variable(torch.zeros(2*self.num_layers, batch_size, self.hidden_dim, device=device))

    def forward(self, cond_data, inp, h):
        """
        inp: batch_size x seq_len x 3 / [type_label, ho_pos]: batch_size x seq_len, batch_size x seq_len x 2
        """
        type_label, ho_pos = inp
        type_label = type_label.permute(1, 0)
        ho_pos = ho_pos.permute(1, 0, 2)
        total_len, batch_size = type_label.shape
        cond_data = cond_data.permute(1, 0, 2)
        # 1 x total_len -> batch_size, total_len
        pos = (torch.arange(total_len, device=cond_data.device, dtype=torch.long) % self.seq_len).reshape([total_len, 1]).expand_as(type_label)

        emb = self.embeddings(type_label)                               # seq_len x batch_size x embedding_dim
        pos_emb = self.pos_embeddings(pos)                               # seq_len x batch_size x embedding_dim
        ho_pos_emb = self.ho_pos_embeddings(ho_pos.reshape([-1, 2])).reshape([total_len, batch_size, -1])  # seq_len x batch_size x embedding_dim

        _, h = self.gru(torch.cat([emb, pos_emb, ho_pos_emb, cond_data], dim=2), h)  # 4 x batch_size x hidden_dim
        h = h.permute(1, 0, 2).contiguous()  # batch_size x 4 x hidden_dim

        out = self.gru2hidden(h.view(-1, 2*self.num_layers*self.hidden_dim))  # batch_size x 4*hidden_dim
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
        if not isinstance(inp, (list, tuple)):
            inp = (inp[:, :, 0].long(), inp[:, :, 1:])
        if h is None:
            h = self.init_hidden(inp[0].shape[0], device=cond_data.device)
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

