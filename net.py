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
