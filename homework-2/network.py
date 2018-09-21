import torch
from torch import nn


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(1, 2, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear = nn.Sequential(
            nn.Linear(7 * 7 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.convolution(x)
        x = x.view(-1, 7 * 7 * 4)
        x = self.linear(x)

        if self.training:
            return x

        return self.softmax(x)