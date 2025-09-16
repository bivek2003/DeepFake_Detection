"""
Base Model Infrastructure for Deepfake Detection

Provides abstract base classes and common functionality for all deepfake detection models.
Supports both video and audio modalities with consistent interfaces.

Author: Bivek Sharma Panthi
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for deepfake detection models"""
    model_name: str
    input_size: Union[Tuple[int, int], int]  # (H, W) for images or length for audio
    num_classes: int = 2
    dropout_rate: float = 0.2
    use_pretrained: bool = True
    freeze_backbone: bool = False
    
    # Training specific
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    
    # Model specific parameters
    model_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {}


@dataclass
class ModelOutput:
    """Standardized model output"""
    logits: torch.Tensor  # Raw model outputs
    probabilities: torch.Tensor  # Softmax probabilities
    predictions: torch.Tensor  # Predicted classes
    features: Optional[torch.Tensor] = None  # Intermediate features
    attention_weights: Optional[torch.Tensor] = None  # Attention maps if available


class BaseDeepfakeModel(nn.Module, ABC):
    """Abstract base class for all deepfake detection models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.num_classes = config.num_classes
        self.input_size = config.input_size
        
        # Model metadata
        self.created_at = time.time()
        self.training_history = []
        self.best_accuracy = 0.0
        self.total_params = 0
        self.trainable_params = 0
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    @abstractmethod
    def build_backbone(self) -> nn.Module:
        """Build the backbone architecture (e.g., EfficientNet, AASIST)"""
        pass
    
    @abstractmethod
    def build_classifier(self, backbone_features: int) -> nn.Module:
        """Build the classification head"""
        pass
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, ModelOutput]:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor
            return_features: Whether to return detailed output with features
            
        Returns:
            Raw logits or detailed ModelOutput
        """
        # Extract features
        features = self.extract_features(x)
        
        # Classification
        logits = self.classify(features)
        
        if return_features or self.training:
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            return ModelOutput(
                logits=logits,
                probabilities=probabilities,
                predictions=predictions,
                features=features
            )
        
        return logits
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using the backbone"""
        pass
    
    @abstractmethod
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify features using the classification head"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        # Calculate parameter counts
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "model_class": self.__class__.__name__,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,
            "model_size_mb": self.total_params * 4 / (1024 * 1024),  # Assuming float32
            "config": self.config.__dict__,
            "created_at": self.created_at,
            "best_accuracy": self.best_accuracy
        }
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen for fine-tuning")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = True
            logger.info("Backbone unfrozen")
    
    def get_lr_schedule_params(self) -> Dict[str, Any]:
        """Get recommended learning rate schedule parameters"""
        return {
            "initial_lr": self.config.learning_rate,
            "warmup_epochs": 5,
            "scheduler_type": "cosine",
            "min_lr": self.config.learning_rate * 0.01
        }


class VideoDeepfakeModel(BaseDeepfakeModel):
    """Base class for video-based deepfake detection models"""
    
    def __init__(self, config: ModelConfig):
        # Validate input size for video
        if not isinstance(config.input_size, tuple) or len(config.input_size) != 2:
            raise ValueError("Video models require input_size as (height, width) tuple")
        
        super().__init__(config)
        self.height, self.width = config.input_size
        self.channels = 3  # RGB
        
        # Build model
        self.backbone = self.build_backbone()
        backbone_features = self.get_backbone_features()
        self.classifier = self.build_classifier(backbone_features)
        
        logger.info(f"Video model initialized: {self.height}x{self.width} input")
    
    @abstractmethod
    def get_backbone_features(self) -> int:
        """Get number of features from backbone"""
        pass
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from video frames"""
        # x shape: (batch_size, channels, height, width)
        return self.backbone(x)
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify video features"""
        return self.classifier(features)


class AudioDeepfakeModel(BaseDeepfakeModel):
    """Base class for audio-based deepfake detection models"""
    
    def __init__(self, config: ModelConfig):
        # Validate input size for audio
        if not isinstance(config.input_size, (int, tuple)):
            raise ValueError("Audio models require input_size as int (length) or tuple")
        
        super().__init__(config)
        
        if isinstance(config.input_size, int):
            self.input_length = config.input_size
            self.input_features = 1  # Raw audio
        else:
            self.input_features, self.input_length = config.input_size  # (features, time)
        
        # Build model
        self.backbone = self.build_backbone()
        backbone_features = self.get_backbone_features()
        self.classifier = self.build_classifier(backbone_features)
        
        logger.info(f"Audio model initialized: {self.input_features}x{self.input_length} input")
    
    @abstractmethod
    def get_backbone_features(self) -> int:
        """Get number of features from backbone"""
        pass
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from audio"""
        # x shape: (batch_size, features, time) or (batch_size, time)
        return self.backbone(x)
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify audio features"""
        return self.classifier(features)


class ModelRegistry:
    """Registry for available deepfake detection models"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
    
    def register_model(self, name: str, model_class, default_config: ModelConfig):
        """Register a model class with default configuration"""
        self.models[name] = model_class
        self.model_configs[name] = default_config
        logger.info(f"Registered model: {name}")
    
    def create_model(self, name: str, config: Optional[ModelConfig] = None) -> BaseDeepfakeModel:
        """Create model instance by name"""
        if name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        
        model_class = self.models[name]
        
        if config is None:
            config = self.model_configs[name]
        
        return model_class(config)
    
    def list_models(self) -> Dict[str, str]:
        """List all registered models"""
        return {name: cls.__name__ for name, cls in self.models.items()}
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get information about a registered model"""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found")
        
        model_class = self.models[name]
        default_config = self.model_configs[name]
        
        return {
            "name": name,
            "class": model_class.__name__,
            "modality": "video" if issubclass(model_class, VideoDeepfakeModel) else "audio",
            "default_config": default_config.__dict__
        }


# Global model registry
model_registry = ModelRegistry()


class ModelFactory:
    """Factory for creating deepfake detection models"""
    
    @staticmethod
    def create_video_model(model_name: str, 
                          input_size: Tuple[int, int] = (224, 224),
                          **kwargs) -> VideoDeepfakeModel:
        """Create a video deepfake detection model"""
        config = ModelConfig(
            model_name=model_name,
            input_size=input_size,
            **kwargs
        )
        
        return model_registry.create_model(model_name, config)
    
    @staticmethod
    def create_audio_model(model_name: str,
                          input_size: Union[int, Tuple[int, int]] = 48000,
                          **kwargs) -> AudioDeepfakeModel:
        """Create an audio deepfake detection model"""
        config = ModelConfig(
            model_name=model_name,
            input_size=input_size,
            **kwargs
        )
        
        return model_registry.create_model(model_name, config)
    
    @staticmethod
    def list_available_models() -> Dict[str, Dict[str, Any]]:
        """List all available models with their information"""
        models_info = {}
        for name in model_registry.list_models():
            try:
                models_info[name] = model_registry.get_model_info(name)
            except Exception as e:
                logger.warning(f"Error getting info for model {name}: {e}")
        
        return models_info


def load_model_weights(model: BaseDeepfakeModel, 
                      checkpoint_path: Union[str, Path],
                      device: torch.device = None) -> BaseDeepfakeModel:
    """Load model weights from checkpoint"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load training history if available
    if 'training_history' in checkpoint:
        model.training_history = checkpoint['training_history']
    
    if 'best_accuracy' in checkpoint:
        model.best_accuracy = checkpoint['best_accuracy']
    
    logger.info(f"Loaded model weights from {checkpoint_path}")
    return model


def save_model_checkpoint(model: BaseDeepfakeModel,
                         checkpoint_path: Union[str, Path],
                         optimizer_state: Optional[Dict] = None,
                         epoch: Optional[int] = None,
                         loss: Optional[float] = None,
                         accuracy: Optional[float] = None):
    """Save model checkpoint with metadata"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_info': model.get_model_info(),
        'training_history': model.training_history,
        'best_accuracy': model.best_accuracy,
        'save_time': time.time()
    }
    
    if optimizer_state:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
        if accuracy > model.best_accuracy:
            model.best_accuracy = accuracy
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved model checkpoint to {checkpoint_path}")


def main():
    """Demonstrate base model infrastructure"""
    print("🤖 BASE MODEL INFRASTRUCTURE DEMO")
    print("=" * 50)
    
    # Show model registry
    factory = ModelFactory()
    available_models = factory.list_available_models()
    
    print(f"📋 Available Models: {len(available_models)}")
    for name, info in available_models.items():
        print(f"  • {name}: {info.get('class', 'Unknown')} ({info.get('modality', 'unknown')})")
    
    # Demonstrate model config
    video_config = ModelConfig(
        model_name="efficientnet_b4",
        input_size=(224, 224),
        num_classes=2,
        dropout_rate=0.2,
        learning_rate=1e-4
    )
    
    audio_config = ModelConfig(
        model_name="aasist",
        input_size=48000,  # 3 seconds at 16kHz
        num_classes=2,
        dropout_rate=0.3,
        learning_rate=5e-5
    )
    
    print(f"\n⚙️ Example Configurations:")
    print(f"  Video Config: {video_config.model_name} - {video_config.input_size}")
    print(f"  Audio Config: {audio_config.model_name} - {audio_config.input_size}")
    
    print(f"\n✅ Base model infrastructure ready!")
    print(f"🎯 Ready for specific model implementations")


if __name__ == "__main__":
    main()
