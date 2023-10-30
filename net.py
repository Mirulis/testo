from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional


class LeNet(nn.Module):
    def __init__(self, *features: int, dropout: float = 0):
        super().__init__()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.fc = nn.ModuleList(
            nn.Linear(features[i], features[i + 1]) for i in range(len(features) - 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for linear in self.fc[:-1]:
            x = functional.tanh(self.dropout(linear(x)))
        return self.softmax(self.fc[-1](x))


class TaylorSeries(nn.Module):
    def __init__(self, n: int, *features: int, dropout: float = 0,
                 activation: Callable[[torch.Tensor], torch.Tensor] = None):
        super().__init__()
        self.exponent = torch.arange(0, n, dtype=torch.int)

        linear_features = [n]
        linear_features.extend(features)
        linear_features.append(1)

        self.fc = nn.ModuleList(
            nn.Linear(linear_features[i], linear_features[i + 1]) for i in range(len(linear_features) - 1)
        )

        self.activation = activation or (lambda t: t)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        x = torch.pow(x.unsqueeze(-1), self.exponent)
        for linear in self.fc[:-1]:
            x = self.activation(self.dropout(linear(x)))
        return self.fc[-1](x).squeeze(-1)
