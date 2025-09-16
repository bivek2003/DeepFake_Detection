"""
Models module for deepfake detection

This module provides model architectures for both video and audio deepfake detection,
following the roadmap specifications for state-of-the-art performance.

Supported Models:
- Video: EfficientNet, XceptionNet, Vision Transformers
- Audio: AASIST, Wav2Vec2+AASIST, RawNet2
- Ensemble: Multi-modal fusion methods

Author: Your Name
"""

from .base_model import (
    BaseDeepfakeModel,
    VideoDeepfakeModel, 
    AudioDeepfakeModel,
    ModelConfig,
    ModelOutput,
    ModelRegistry,
    ModelFactory,
    model_registry,
    load_model_weights,
    save_model_checkpoint
)

__all__ = [
    # Base classes
    "BaseDeepfakeModel",
    "VideoDeepfakeModel",
    "AudioDeepfakeModel",
    
    # Configuration and output
    "ModelConfig", 
    "ModelOutput",
    
    # Model management
    "ModelRegistry",
    "ModelFactory", 
    "model_registry",
    
    # Utilities
    "load_model_weights",
    "save_model_checkpoint",
]

# Version info
__version__ = "0.1.0"
