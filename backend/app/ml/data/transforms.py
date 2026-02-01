"""
Data augmentation transforms for deepfake detection training.

Key augmentations for robust deepfake detection:
- Compression artifacts simulation (JPEG quality)
- Blur to handle different video qualities
- Color jitter for lighting variations
- Geometric transforms for pose variations
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, Any, Optional


# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 380,
    use_heavy_augmentation: bool = True
) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size (EfficientNet-B4 uses 380)
        use_heavy_augmentation: Whether to use aggressive augmentation
    
    Returns:
        Albumentations Compose object
    """
    transforms_list = [
        # Resize and crop
        A.Resize(image_size + 32, image_size + 32),
        A.RandomCrop(image_size, image_size),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=(-15, 15),
            mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
    ]
    
    if use_heavy_augmentation:
        transforms_list.extend([
            # Color augmentations
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
            ], p=0.5),
            
            # Quality degradation (critical for deepfake detection)
            A.OneOf([
                # JPEG compression artifacts
                A.ImageCompression(quality_range=(30, 100), p=1.0),
                # Gaussian blur
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                # Downscale then upscale (simulates low quality)
                A.Downscale(scale_range=(0.5, 0.9), p=1.0),
            ], p=0.5),
            
            # Noise
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.3),
            
            # Cutout / Random erasing
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(image_size // 16, image_size // 8),
                hole_width_range=(image_size // 16, image_size // 8),
                fill="constant",
                fill_value=0,
                p=0.3
            ),
        ])
    
    # Normalize and convert to tensor
    transforms_list.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    return A.Compose(transforms_list)


def get_val_transforms(image_size: int = 380) -> A.Compose:
    """
    Get validation/testing augmentation pipeline (minimal transforms).
    
    Args:
        image_size: Target image size
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_test_transforms(image_size: int = 380) -> A.Compose:
    """Alias for validation transforms."""
    return get_val_transforms(image_size)


def get_tta_transforms(image_size: int = 380) -> list:
    """
    Get Test-Time Augmentation transforms for inference.
    
    Returns list of transforms to apply, predictions are averaged.
    """
    base_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    flip_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    
    return [base_transform, flip_transform]


class DeepfakeAugmentation:
    """
    Specialized augmentations that simulate deepfake artifacts.
    """
    
    def __init__(self, p: float = 0.3):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply deepfake-specific augmentation."""
        if np.random.random() > self.p:
            return image
        
        augmentation = np.random.choice([
            self._blend_boundary,
            self._color_mismatch,
            self._face_warping,
        ])
        
        return augmentation(image)
    
    def _blend_boundary(self, image: np.ndarray) -> np.ndarray:
        """Simulate face blending boundary artifacts."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 3
        
        # Create elliptical mask
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, center, (radius, int(radius * 1.3)), 0, 0, 360, 1, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Slightly shift colors inside mask
        shift = np.random.randint(-10, 10, 3)
        shifted = image.astype(np.int32) + shift
        shifted = np.clip(shifted, 0, 255).astype(np.uint8)
        
        mask = mask[:, :, np.newaxis]
        result = (image * (1 - mask * 0.3) + shifted * mask * 0.3).astype(np.uint8)
        
        return result
    
    def _color_mismatch(self, image: np.ndarray) -> np.ndarray:
        """Simulate color mismatch between face and background."""
        # Random color temperature shift
        image = image.astype(np.float32)
        
        # Warm or cool shift
        if np.random.random() > 0.5:
            image[:, :, 0] *= 1 + np.random.uniform(0.02, 0.08)  # Red
            image[:, :, 2] *= 1 - np.random.uniform(0.02, 0.08)  # Blue
        else:
            image[:, :, 0] *= 1 - np.random.uniform(0.02, 0.08)
            image[:, :, 2] *= 1 + np.random.uniform(0.02, 0.08)
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def _face_warping(self, image: np.ndarray) -> np.ndarray:
        """Simulate subtle face warping artifacts."""
        h, w = image.shape[:2]
        
        # Create subtle distortion grid
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Add subtle sinusoidal distortion
        freq = np.random.uniform(0.01, 0.03)
        amp = np.random.uniform(1, 3)
        
        grid_x = grid_x + amp * np.sin(freq * grid_y)
        grid_y = grid_y + amp * np.sin(freq * grid_x)
        
        grid_x = np.clip(grid_x, 0, w - 1).astype(np.float32)
        grid_y = np.clip(grid_y, 0, h - 1).astype(np.float32)
        
        result = cv2.remap(image, grid_x, grid_y, cv2.INTER_LINEAR)
        
        return result
