"""
EfficientNet models for deepfake detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any
import warnings

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

class EfficientNetDeepfake(BaseDeepfakeModel):
    """EfficientNet-based deepfake detection model."""
    
    def __init__(self, 
                 variant: str = "b4",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.2,
                 config: Optional[ModelConfig] = None):
        """
        Initialize EfficientNet model.
        
        Args:
            variant: EfficientNet variant (b0, b1, ..., b7)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate
            config: Model configuration
        """
        super().__init__(config)
        
        self.variant = variant.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Map variant to torchvision model
        model_map = {
            'b0': models.efficientnet_b0,
            'b1': models.efficientnet_b1,
            'b2': models.efficientnet_b2,
            'b3': models.efficientnet_b3,
            'b4': models.efficientnet_b4,
            'b5': models.efficientnet_b5,
            'b6': models.efficientnet_b6,
            'b7': models.efficientnet_b7,
        }
        
        if self.variant not in model_map:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}. "
                           f"Supported: {list(model_map.keys())}")
        
        # Create backbone
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = model_map[self.variant](pretrained=pretrained)
        
        # Get feature dimension
        feature_dim = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        # Remove the classifier
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        return features
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'variant': self.variant,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
        }

def create_efficientnet_model(variant: str = "b4",
                             num_classes: int = 2,
                             pretrained: bool = True,
                             dropout_rate: float = 0.2,
                             **kwargs) -> EfficientNetDeepfake:
    """
    Create an EfficientNet model for deepfake detection.
    
    Args:
        variant: EfficientNet variant (b0-b7)
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        dropout_rate: Dropout rate
        **kwargs: Additional arguments
    
    Returns:
        EfficientNetDeepfake model
    """
    config = ModelConfig(
        model_type="efficientnet",
        variant=variant,
        num_classes=num_classes,
        input_size=(3, 224, 224),
        **kwargs
    )
    
    return EfficientNetDeepfake(
        variant=variant,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        config=config
    )

# Convenience functions for specific variants
def create_efficientnet_b4(**kwargs) -> EfficientNetDeepfake:
    """Create EfficientNet-B4 model."""
    return create_efficientnet_model(variant="b4", **kwargs)

def create_efficientnet_b7(**kwargs) -> EfficientNetDeepfake:
    """Create EfficientNet-B7 model."""
    return create_efficientnet_model(variant="b7", **kwargs)

__all__ = [
    'EfficientNetDeepfake',
    'create_efficientnet_model',
    'create_efficientnet_b4',
    'create_efficientnet_b7'
]
