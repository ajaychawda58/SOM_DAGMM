import torch
from torch import nn


class EstimationNetwork(nn.Module):
    """Defines a estimation network."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 10),
                                 nn.Tanh(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(10, 2),
                                 nn.Softmax(dim=1))

    def forward(self, input):
        return self.net(input)