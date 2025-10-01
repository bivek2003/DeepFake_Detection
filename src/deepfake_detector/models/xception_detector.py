"""
Xception-inspired architecture for deepfake detection
"""

import torch
import torch.nn as nn
from torchvision import models

class XceptionDeepfakeDetector(nn.Module):
    """Xception-inspired deepfake detector using ResNet50 backbone"""
    
    def __init__(self, num_classes=2):
        super(XceptionDeepfakeDetector, self).__init__()
        
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        
        # Xception-style classifier with separable convolutions concept
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
