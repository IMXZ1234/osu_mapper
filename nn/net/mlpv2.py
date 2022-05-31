from torch import nn
from torch.nn import functional as F


class MLPv2(nn.Module):
    def __init__(self,
                 layers=(512+16*3, (512+16*3)*2, 3),
                 ):
        super().__init__()
        self.fc_layers = nn.ModuleList(
            [nn.Linear(n1, n2) for n1, n2 in zip(layers[:-1], layers[1:])]
        )

    def forward(self, x):
        # x = x.reshape([x.shape[0], -1])
        for fc_layer in self.fc_layers[:-1]:
            x = fc_layer(x)
            x = F.relu(x, inplace=True)
        x = self.fc_layers[-1](x)
        return x
