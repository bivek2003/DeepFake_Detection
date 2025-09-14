"""
Data handling module for deepfake detection

This module provides functionality for:
- Dataset management and organization
- Data preprocessing for video and audio
- Data loading and augmentation pipelines
- Train/validation/test splitting

Author: Bivek Sharma Panthi
"""

from .dataset_manager import DatasetManager, DatasetRegistry, DatasetInfo

__all__ = [
    "DatasetManager",
    "DatasetRegistry", 
    "DatasetInfo",
]
