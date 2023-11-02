from __future__ import annotations

import torch
from torch import nn, device
from torch.nn import functional
from collections.abc import Callable


class LeNet(nn.Module):
    def __init__(self, *features: int, dropout: float = 0):
        super().__init__()

        self.fs = FourierSeries(11, n=10000)

        # self.ts = SeparatedTaylorSeries(
        #     10, dropout=dropout, n_var=11, activation=functional.tanh
        # )

        self.fc = nn.ModuleList(
            nn.Linear(features[i], features[i + 1]) for i in range(len(features) - 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        x = self.fs(x)
        # x = self.ts(functional.tanh(x))
        for linear in self.fc[:-1]:
            x = functional.tanh(self.dropout(linear(x)))
        return self.softmax(self.fc[-1](x))


class TaylorSeries(nn.Module):
    def __init__(
        self,
        n: int,
        *features: int,
        dropout: float = 0,
        n_var: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = lambda t: t,
    ):
        super().__init__()
        self.exponent = torch.arange(0, n, dtype=torch.int)

        linear_features = [n_var * n]
        linear_features.extend(n_var * f for f in features)
        linear_features.append(n_var)

        self.fc = nn.ModuleList(
            nn.Linear(linear_features[i], linear_features[i + 1])
            for i in range(len(linear_features) - 1)
        )

        self.activation = activation or (lambda t: t)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        is_batch = len(x.shape) > 1

        if self.exponent.device != x.device:
            self.exponent = self.exponent.to(x.device)

        x = torch.pow(x.unsqueeze(-1), self.exponent)
        if is_batch:
            x = x.view(x.size(0), -1)
        else:
            x = x.view(-1)

        for linear in self.fc[:-1]:
            x = self.activation(self.dropout(linear(x)))
        x = self.fc[-1](x)

        if is_batch:
            return x
        else:
            return x.view(x.size(0))


class SeparatedTaylorSeries(nn.Module):
    def __init__(
        self,
        n: int,
        *features: int,
        dropout: float = 0,
        n_var: int = 1,
        activation: Callable[[torch.Tensor], torch.Tensor] = lambda t: t,
    ):
        super().__init__()
        self.tss = nn.ModuleList(
            TaylorSeries(n, *features, dropout=dropout, activation=activation)
            for _ in range(n_var)
        )

    def forward(self, x):
        if len(x.shape) == 1:
            return torch.cat([ts(x[i]) for i, ts in enumerate(self.tss)])
        return torch.stack(
            [ts(x[:, i].unsqueeze(-1)).view(-1) for i, ts in enumerate(self.tss)], dim=1
        )


class FourierSeries(nn.Module):
    def __init__(
        self,
        n_var: int = 1,
        n: int = 10,
        period: float = 2,
        device: device | str | None = None,
    ):
        super().__init__()
        cst = 2 * torch.pi / period
        self.n_var = n_var
        self.range = torch.arange(1, n + 1, dtype=torch.float32, device=device) * cst
        self.sin_params = nn.Parameter(torch.randn(n * n_var, device=device), True)
        self.cos_params = nn.Parameter(torch.randn(n * n_var, device=device), True)
        self.bias = nn.Parameter(torch.randn(1, device=device), True)

    def forward(self, x: torch.Tensor):
        if self.range.device != x.device:
            self.range = self.range.to(x.device)
        if self.sin_params.device != x.device:
            self.sin_params = self.sin_params.to(x.device)
        if self.cos_params.device != x.device:
            self.cos_params = self.cos_params.to(x.device)
        if self.bias.device != x.device:
            self.bias = self.bias.to(x.device)

        if self.n_var == 1:
            x = x.unsqueeze(-1) * self.range
            return (
                (torch.sin(x) * self.sin_params).sum(-1)
                + (torch.cos(x) * self.cos_params).sum(-1)
                + self.bias
            )
        else:
            if len(x.shape) == 1:
                x = (x.unsqueeze(-1) * self.range).view(-1)
                return (
                    (torch.sin(x) * self.sin_params).sum(-1)
                    + (torch.cos(x) * self.cos_params).sum(-1)
                    + self.bias
                )
            else:
                batch_size = x.size(0)
                x = (x.unsqueeze(-1) * self.range).view(batch_size, -1)
                return (
                    (torch.sin(x) * self.sin_params)
                    .view(batch_size, self.n_var, -1)
                    .sum(-1)
                    + (torch.cos(x) * self.cos_params)
                    .view(batch_size, self.n_var, -1)
                    .sum(-1)
                    + self.bias
                )


class SeparatedFourierSeries(nn.Module):
    def __init__(
        self,
        n_var: int,
        n: int = 10,
        period: float = 2,
        device: device | str | None = None,
    ):
        super().__init__()
        self.fss = nn.ModuleList(
            FourierSeries(1, n, period, device) for _ in range(n_var)
        )

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 1:
            return torch.cat([fs(x[i]) for i, fs in enumerate(self.fss)])
        return torch.stack(
            [fs(x[:, i].unsqueeze(-1)).view(-1) for i, fs in enumerate(self.fss)],
            dim=-1,
        )
