import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    """Configuration class for models"""
    def __init__(self, num_classes: int = 2, input_size: Tuple[int, int] = (224, 224), 
                 dropout: float = 0.2, **kwargs):
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout = dropout
        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class BaseDeepfakeModel(nn.Module, ABC):
    """Base class for all deepfake detection models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        
        # Build the model components
        self.backbone = self.build_backbone()
        self.classifier = self.build_classifier()
        
    @abstractmethod
    def build_backbone(self) -> nn.Module:
        """Build the backbone architecture"""
        pass
        
    @abstractmethod  
    def build_classifier(self) -> nn.Module:
        """Build the classification head"""
        pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        # Flatten features if needed
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)
        
    def classify(self, x: torch.Tensor) -> Dict[str, Any]:
        """Classify input with confidence scores"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
        return {
            'predicted_class': predicted_class.cpu().numpy(),
            'confidence': confidence.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy()
        }
