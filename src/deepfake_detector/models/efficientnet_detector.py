#!/usr/bin/env python3
"""
EfficientNet-based deepfake detector
"""

import torch
import torch.nn as nn
from torchvision import models

class EfficientNetDeepfakeDetector(nn.Module):
    """EfficientNet-based deepfake detector"""
    
    def __init__(self, model_name='efficientnet_b4', num_classes=2, pretrained=True):
        super(EfficientNetDeepfakeDetector, self).__init__()
        
        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
        else:
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
