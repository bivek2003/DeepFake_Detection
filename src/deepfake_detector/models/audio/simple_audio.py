import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAudioModel(nn.Module):
    def __init__(self, n_mels: int = 80, num_classes: int = 2):
        super().__init__()
        # Use frequency bins (80) as input channels
        self.conv1 = nn.Conv1d(in_channels=n_mels, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Expected input: [batch, 1, n_mels, time]
        if x.dim() == 4:
            x = x.squeeze(1)   # [batch, n_mels, time]
        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))  # [batch, 128, time]
        x = F.relu(self.bn2(self.conv2(x)))  # [batch, 256, time]
        # Pool across time
        x = self.pool(x).squeeze(-1)         # [batch, 256]
        # Classification
        return self.fc(x)

