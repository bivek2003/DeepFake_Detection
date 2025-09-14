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
from .video_processor import VideoProcessor, FaceDetector, VideoMetadata, FaceDetection
from .audio_processor import AudioProcessor, AudioMetadata, AudioFeatures
from .data_pipeline import (
    DeepfakeVideoDataset,
    DeepfakeAudioDataset,
    DataSplitter,
    DataSplit,
    VideoAugmentations,
    AudioAugmentations,
    DataPipelineManager
)

__all__ = [
    # Dataset management
    "DatasetManager",
    "DatasetRegistry", 
    "DatasetInfo",
    
    # Video processing
    "VideoProcessor",
    "FaceDetector",
    "VideoMetadata",
    "FaceDetection",
    
    # Audio processing
    "AudioProcessor",
    "AudioMetadata", 
    "AudioFeatures",
    
    # PyTorch pipeline
    "DeepfakeVideoDataset",
    "DeepfakeAudioDataset",
    "DataSplitter",
    "DataSplit",
    "VideoAugmentations",
    "AudioAugmentations", 
    "DataPipelineManager",
]
