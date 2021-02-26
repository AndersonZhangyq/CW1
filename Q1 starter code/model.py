import torch
from torch import nn


class FCModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = x.flatten(1)
        output = self.fc(x)
        return output
