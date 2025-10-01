"""
EfficientNet-based deepfake detector
"""

import torch
import torch.nn as nn
from torchvision import models

class EfficientNetDeepfakeDetector(nn.Module):
    """EfficientNet-based deepfake detector optimized for face detection"""
    
    def __init__(self, model_name='efficientnet_b4', num_classes=2, pretrained=True):
        super(EfficientNetDeepfakeDetector, self).__init__()
        
        if model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
        else:
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
        
        # Enhanced classifier for deepfake detection
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
