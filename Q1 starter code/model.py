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


class CNNModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2),
                                     nn.Conv2d(64, 128, kernel_size=3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2),
                                     nn.Conv2d(128, 256, kernel_size=3),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3, stride=2))
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(nn.Linear(256 * 7 * 7, 1024),
                                        nn.ReLU(), nn.Dropout(),
                                        nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.feature(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        output = self.classifier(x)
        return output
