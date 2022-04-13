"""Defines the compression network."""

import torch
from torch import nn


class CompressionNetwork(nn.Module ):
    """Defines a compression network."""
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.encoder = nn.Sequential(nn.Linear(self.size, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 10),
                                     nn.Tanh(),
                                     nn.Linear(10, self.size))

        self._reconstruction_loss = nn.MSELoss()

    def forward(self, input):
        out = self.encoder(input)
        out = self.decoder(out)

        return out

    def encode(self,  input):
        return self.encoder(input)

    def decode(self, input):
        return self.decoder(input)

    def reconstruction_loss(self, input):
        target_hat = self(input)
        return self._reconstruction_loss(target_hat, input)