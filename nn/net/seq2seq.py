import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def gen_pos_seq(batch_size, seq_len, period=8, device='cpu'):
    return (torch.arange(seq_len, device=device, dtype=torch.long) % period).repeat(batch_size, 1)


class Encoder(nn.Module):
    def __init__(self, cond_data_feature_dim, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.cond_data_feature_dim = cond_data_feature_dim
        self.hidden_size = hidden_size
        self.gru = nn.GRU(cond_data_feature_dim, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        # L, N, C = src.shape
        outputs, hidden = self.gru(src, hidden)
        # sum bidirectional outputs
        # L, N, self.hidden_size
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # hidden: 2*n_layers, N, self.hidden_size
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        # hidden_size
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # encoder_output: L, N, self.hidden_size
        timestep = encoder_outputs.size(0)
        # N, self.hidden_size -> L, N, self.hidden_size -> N, L, self.hidden_size
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        # N, L, self.hidden_size
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # N, L, 2*self.hidden_size -> N, L, self.hidden_size
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [N, self.hidden_size, L]
        # hidden_size, -> N, hidden_size -> N, 1, hidden_size
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)  # N, 1, L
        return energy.squeeze(1)  # N, L


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        """
        hidden_size:
        """
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        # hidden from encoder + type_label_emb, ho_pos_emb, pos_emb
        self.gru = nn.GRU(hidden_size + embed_size * 3, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inp, last_hidden, encoder_outputs):
        # input is last label
        # last hidden is hidden representation learned by encoder: [self.decoder.n_layers, N, self.hidden_size]
        # Get the embedding of the current input word (last output word)
        # 1, N, self.emb_size
        embedded = inp.unsqueeze(0)
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        # N, L
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        # N, L * N, L, self.hidden_size -> N, 1, self.hidden_size
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # 1, N, self.hidden_size
        context = context.transpose(0, 1)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)
        # 1, N, self.hidden_size -> N, self.hidden_size
        output = output.squeeze(0)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], dim=1))
        # output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder_args, decoder_args, num_class, embed_size, period=8):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(**encoder_args)
        self.decoder = Decoder(**decoder_args)
        self.type_label_embed = nn.Embedding(num_class, embed_size)
        self.pos_embed = nn.Embedding(period, embed_size)
        self.ho_pos_embed = nn.Linear(2, embed_size)
        self.period = period
        self.num_class = num_class

    def embed_all(self, type_label, ho_pos):
        type_label = self.type_label_embed(type_label)
        ho_pos = self.ho_pos_embed(ho_pos)
        return type_label, ho_pos

    def default_start_label(self, batch_size, device='cpu'):
        output = torch.tensor([0, 0.5, 0.5], device=device).repeat(batch_size, 1)  # sos
        type_label, ho_pos = output[:, 0].long(), output[:, 1:]
        type_label, ho_pos = self.embed_all(type_label, ho_pos)
        pos_emb = self.pos_embed(torch.tensor([self.period-1], dtype=torch.long, device=device)).repeat(batch_size, 1)
        output = torch.cat([type_label, ho_pos, pos_emb], dim=-1)
        return output

    def forward(self, cond_data, label, teacher_forcing_ratio=0.5, last_label=None):
        """
        src: cond_data
        trg: label
        """
        # N, L, C -> L, N, C
        batch_size, seq_len, _ = cond_data.shape
        outputs = []

        pos_seq = gen_pos_seq(batch_size, seq_len, self.period, cond_data.device)
        # N, L -> N, L, embed_size
        pos_emb = self.pos_embed(pos_seq)

        type_label, ho_pos = label[:, :, 0].long(), label[:, :, 1:]
        type_label, ho_pos = self.embed_all(type_label, ho_pos.reshape([batch_size*seq_len, -1]))
        label = torch.cat([type_label, ho_pos.reshape([batch_size, seq_len, -1]), pos_emb], dim=-1)

        pos_emb = pos_emb.transpose(0, 1)
        cond_data = cond_data.transpose(0, 1)
        label = label.transpose(0, 1)

        # encoder_output: L, N, self.hidden_size
        encoder_output, hidden = self.encoder(cond_data)
        # hidden: 2*self.encoder.n_layers, N, self.hidden_size -> self.decoder.n_layers, N, self.hidden_size
        hidden = hidden[:self.decoder.n_layers]

        if last_label is None:
            output = self.default_start_label(batch_size, cond_data.device)  # sos
        else:
            output = last_label
        for t in range(seq_len):
            # output: N, num_class + 2
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs.append(output)
            is_teacher = random.random() < teacher_forcing_ratio
            # top1 = output.data.max(1)[1]
            if is_teacher:
                output = label[t]
            else:
                # type_label: N, num_class
                # ho_pos: N, 2
                type_label, ho_pos = output[:, :self.num_class], output[:, self.num_class:]
                type_label = torch.argmax(type_label, dim=1)
                type_label, ho_pos = self.embed_all(type_label, ho_pos)
                output = torch.cat([type_label, ho_pos, pos_emb[t]], dim=1)
                # -> N, L, C
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def generate(self, cond_data, last_label=None):
        """
        src: cond_data
        trg: label
        """
        # N, L, C -> L, N, C
        batch_size, seq_len, _ = cond_data.shape
        outputs = []

        pos_seq = gen_pos_seq(batch_size, seq_len, self.period, cond_data.device)
        # N, L -> N, L, embed_size
        pos_emb = self.pos_embed(pos_seq)

        pos_emb = pos_emb.transpose(0, 1)
        cond_data = cond_data.transpose(0, 1)

        # encoder_output: L, N, self.hidden_size
        encoder_output, hidden = self.encoder(cond_data)
        # hidden: 2*self.encoder.n_layers, N, self.hidden_size -> self.decoder.n_layers, N, self.hidden_size
        hidden = hidden[:self.decoder.n_layers]

        if last_label is None:
            output = self.default_start_label(batch_size, cond_data.device)  # sos
        else:
            output = last_label
        for t in range(seq_len):
            # output: N, num_class + 2
            output, hidden, attn_weights = self.decoder(output, hidden, encoder_output)
            outputs.append(output)
            # type_label: N, num_class
            # ho_pos: N, 2
            type_label, ho_pos = output[:, :self.num_class], output[:, self.num_class:]
            type_label = torch.argmax(type_label, dim=1)
            type_label, ho_pos = self.embed_all(type_label, ho_pos)
            output = torch.cat([type_label, ho_pos, pos_emb[t]], dim=1)
            # -> N, L, C
        outputs = torch.stack(outputs, dim=1)
        return outputs
