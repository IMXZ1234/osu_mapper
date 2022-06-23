from torch import nn
import torch
from torch.nn import functional as F


class RNNv2Seg(nn.Module):
    """The RNN model."""

    def __init__(self, num_classes=3, num_layers=4, in_channels=512, step_snaps=12*8, **kwargs):
        super(RNNv2Seg, self).__init__()
        self.in_channels = in_channels
        self.output_channels = num_classes * step_snaps
        self.num_hiddens = self.output_channels * 32
        self.fcn_hiddens = self.output_channels * 32
        self.num_classes = num_classes
        self.step_snaps = step_snaps
        self.rnn = nn.RNN(self.in_channels, self.num_hiddens, num_layers=num_layers, batch_first=True)
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.fcn_hiddens)
            self.linear2 = nn.Linear(self.fcn_hiddens, self.output_channels)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.fcn_hiddens)
            self.linear2 = nn.Linear(self.fcn_hiddens, self.output_channels)

    def forward(self, data, state):
        seq_data = data
        # print(seq_data.shape)
        # N, subseq_len, features = seq_data.shape
        # seq_data.shape = [N, subseq_len, features]
        # print('seq_data.shape')
        # print(seq_data.shape)
        y, state = self.rnn(seq_data, state)
        # y.shape = [N, subseq_len, num_hiddens]
        # print('y.shape')
        # print(y.shape)
        # print('state.shape')
        # print(state.shape)
        # we first change the shape of `y` to (`batch_size` * `subseq_len`, `num_hiddens`).
        # output shape is (`batch_size` * `subseq_len`, `num_classes`).
        y = self.linear(y.reshape([-1, y.shape[-1]]))
        y = F.relu(y, inplace=True)
        output = self.linear2(y).reshape([seq_data.shape[0], self.step_snaps, self.num_classes])
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size,
                self.num_hiddens),
                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (
                torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size,
                    self.num_hiddens), device=device),
                torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size,
                    self.num_hiddens), device=device)
            )
