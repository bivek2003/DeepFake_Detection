"""
Xception model for deepfake detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

try:
    from .base_model import BaseDeepfakeModel, ModelConfig
except ImportError:
    # Fallback base class
    class BaseDeepfakeModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config
    
    class ModelConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class SeparableConv2d(nn.Module):
    """Separable Convolution Block."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    """Xception Block with skip connections."""
    
    def __init__(self, in_channels, out_channels, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
        
        self.relu = nn.ReLU(inplace=True)
        rep = []
        
        filters = in_channels
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(out_channels))
            filters = out_channels
        
        for i in range(2):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, 1, bias=False))
            rep.append(nn.BatchNorm2d(out_channels))
        
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        
        self.rep = nn.Sequential(*rep)
    
    def forward(self, inp):
        x = self.rep(inp)
        
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        
        x = x + skip
        return x

class XceptionDeepfake(BaseDeepfakeModel):
    """Xception model for deepfake detection."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 dropout_rate: float = 0.2,
                 config: Optional[ModelConfig] = None):
        """
        Initialize Xception model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            config: Model configuration
        """
        super().__init__(config)
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.block1 = XceptionBlock(64, 128, 2, start_with_relu=False)
        self.block2 = XceptionBlock(128, 256, 2)
        self.block3 = XceptionBlock(256, 728, 2)
        
        # Middle flow
        self.middle_blocks = nn.ModuleList([
            XceptionBlock(728, 728, 1) for _ in range(8)
        ])
        
        # Exit flow
        self.block12 = XceptionBlock(728, 1024, 2, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(2048, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        for block in self.middle_blocks:
            x = block(x)
        
        # Exit flow
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.extract_features(x)
        features = self.dropout(features)
        x = self.fc(features)
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
        }

def create_xception_model(num_classes: int = 2,
                         dropout_rate: float = 0.2,
                         **kwargs) -> XceptionDeepfake:
    """
    Create a Xception model for deepfake detection.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        **kwargs: Additional arguments
    
    Returns:
        XceptionDeepfake model
    """
    config = ModelConfig(
        model_type="xception",
        num_classes=num_classes,
        input_size=(3, 224, 224),
        **kwargs
    )
    
    return XceptionDeepfake(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        config=config
    )

__all__ = [
    'XceptionDeepfake',
    'create_xception_model',
    'SeparableConv2d',
    'XceptionBlock'
]
