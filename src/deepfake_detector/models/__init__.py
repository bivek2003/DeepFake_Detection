"""
Model architectures for deepfake detection - Phase 2
"""

from .efficientnet_detector import EfficientNetDeepfakeDetector
from .xception_detector import XceptionDeepfakeDetector
from .model_trainer import DeepfakeTrainer
from .model_evaluator import ComprehensiveEvaluator

__all__ = [
    'EfficientNetDeepfakeDetector',
    'XceptionDeepfakeDetector', 
    'DeepfakeTrainer',
    'ComprehensiveEvaluator'
]
