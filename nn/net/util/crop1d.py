from torch import nn


class Crop1D(nn.Module):
    """
    Take keep_len items in the center at dim L.
    input: N C L
    output: N C keep_len
    """
    def __init__(self, keep_len):
        super(Crop1D, self).__init__()
        self.keep_len = keep_len

    def forward(self, data):
        ori_len = data.shape[2]
        start_idx = (ori_len - self.keep_len) // 2
        end_idx = self.keep_len + start_idx
        return data[:, :, start_idx:end_idx]
