"""
Models module for deepfake detection

This module provides model architectures for both video and audio deepfake detection,
following the roadmap specifications for state-of-the-art performance.

Supported Models:
- Video: EfficientNet, XceptionNet, Vision Transformers
- Audio: AASIST, Wav2Vec2+AASIST, RawNet2
- Training: Comprehensive training pipeline with mixed precision
- Evaluation: Cross-dataset evaluation and explainability

Author: Your Name
"""

# Base model infrastructure
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

# Video models
from .video_models import (
    EfficientNetDeepfake,
    XceptionNetDeepfake,
    MultiScaleVideoModel,
    create_efficientnet_model,
    create_xception_model
)

# Audio models  
from .audio_models import (
    AASIST,
    Wav2VecAASIST,
    RawNetDeepfake,
    create_aasist_model,
    create_wav2vec_aasist_model,
    create_rawnet_model
)

# Training pipeline
from .training import (
    TrainingConfig,
    DeepfakeTrainer,
    EnsembleTrainer,
    CrossValidator,
    LabelSmoothingLoss,
    create_training_config,
    train_model
)

# Evaluation and metrics
from .evaluation import (
    EvaluationResults,
    ModelEvaluator,
    GradCAM,
    PerformanceBenchmark,
    StatisticalTester,
    save_evaluation_results,
    load_evaluation_results,
    create_evaluation_report
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
    
    # Video models
    "EfficientNetDeepfake",
    "XceptionNetDeepfake", 
    "MultiScaleVideoModel",
    "create_efficientnet_model",
    "create_xception_model",
    
    # Audio models
    "AASIST",
    "Wav2VecAASIST",
    "RawNetDeepfake",
    "create_aasist_model",
    "create_wav2vec_aasist_model", 
    "create_rawnet_model",
    
    # Training
    "TrainingConfig",
    "DeepfakeTrainer",
    "EnsembleTrainer",
    "CrossValidator",
    "LabelSmoothingLoss",
    "create_training_config",
    "train_model",
    
    # Evaluation
    "EvaluationResults",
    "ModelEvaluator",
    "GradCAM",
    "PerformanceBenchmark", 
    "StatisticalTester",
    "save_evaluation_results",
    "load_evaluation_results",
    "create_evaluation_report",
]

# Version info
__version__ = "0.2.0"
