import torch
import torch.nn as nn
import torch.nn.functional as F


class ClapCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.pool(x)
        x = F.relu(self.conv3(x))

        x = x.mean(dim=-1)

        x = x.mean(dim=-1)

        x = self.fc(x)
        return x