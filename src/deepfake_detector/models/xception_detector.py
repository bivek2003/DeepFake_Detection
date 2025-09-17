#!/usr/bin/env python3
"""
Xception-like architecture for deepfake detection
"""

import torch
import torch.nn as nn
from torchvision import models

class XceptionDeepfakeDetector(nn.Module):
    """Xception-like architecture for deepfake detection"""
    
    def __init__(self, num_classes=2):
        super(XceptionDeepfakeDetector, self).__init__()
        
        # Use ResNet50 as backbone (closest to Xception in torchvision)
        self.backbone = models.resnet50(pretrained=True)
        num_features = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
