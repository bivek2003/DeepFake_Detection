"""
PyTorch Dataset classes for deepfake detection training.

Supports:
- FaceForensics++ (multiple manipulation methods)
- Celeb-DF v2
- Combined datasets with balanced sampling
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable, Union
import random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from PIL import Image
import albumentations as A


class DeepfakeDataset(Dataset):
    """
    Base dataset class for deepfake detection.
    
    Expects extracted face images organized in directories.
    """
    
    def __init__(
        self,
        root_dir: str,
        split_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        label_map: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            root_dir: Root directory containing face images
            split_file: Optional file listing images and labels
            transform: Albumentations transform
            label_map: Mapping from directory/filename patterns to labels
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_map = label_map or {"real": 0, "fake": 1}
        
        self.samples: List[Tuple[str, int]] = []
        
        if split_file:
            self._load_from_split(split_file)
        else:
            self._scan_directory()
        
        print(f"Loaded {len(self.samples)} samples")
        self._print_stats()
    
    def _load_from_split(self, split_file: str):
        """Load samples from split file."""
        split_path = Path(split_file)
        
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    path = parts[0]
                    label = int(parts[1])
                    
                    full_path = self.root_dir / path
                    if full_path.exists():
                        # Add all images in this path
                        if full_path.is_dir():
                            for img_path in full_path.glob("*.jpg"):
                                self.samples.append((str(img_path), label))
                        else:
                            self.samples.append((str(full_path), label))
    
    def _scan_directory(self):
        """Scan directory structure for samples."""
        # Common patterns
        real_patterns = ["real", "original", "youtube", "Celeb-real", "YouTube-real"]
        fake_patterns = ["fake", "synthesis", "Deepfakes", "Face2Face", 
                        "FaceSwap", "NeuralTextures", "Celeb-synthesis"]
        
        for img_path in self.root_dir.rglob("*.jpg"):
            path_str = str(img_path).lower()
            
            label = None
            for pattern in real_patterns:
                if pattern.lower() in path_str:
                    label = 0
                    break
            
            if label is None:
                for pattern in fake_patterns:
                    if pattern.lower() in path_str:
                        label = 1
                        break
            
            if label is not None:
                self.samples.append((str(img_path), label))
    
    def _print_stats(self):
        """Print dataset statistics."""
        labels = [s[1] for s in self.samples]
        real_count = labels.count(0)
        fake_count = labels.count(1)
        print(f"  Real: {real_count}, Fake: {fake_count}")
        if real_count + fake_count > 0:
            print(f"  Fake ratio: {fake_count / (real_count + fake_count):.2%}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a random valid sample on error
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image, label
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling."""
        labels = [s[1] for s in self.samples]
        class_counts = [labels.count(0), labels.count(1)]
        
        # Inverse frequency weighting
        weights = [1.0 / class_counts[label] for label in labels]
        
        return torch.DoubleTensor(weights)


class CelebDFDataset(DeepfakeDataset):
    """Dataset for Celeb-DF v2."""
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        faces_dir: Optional[str] = None,
    ):
        """
        Initialize Celeb-DF dataset.
        
        Args:
            root_dir: Root directory of datasets
            split: One of 'train', 'val', 'test'
            transform: Albumentations transform
            faces_dir: Directory with extracted faces (default: {root}/faces)
        """
        self.split = split
        
        root_path = Path(root_dir)
        
        # Determine faces directory
        if faces_dir:
            faces_path = Path(faces_dir)
        else:
            faces_path = root_path / "faces" / "Celeb-DF-v2"
        
        # Split file path
        split_file = root_path / "splits" / f"celebdf_{split}.txt"
        
        if split_file.exists():
            super().__init__(
                root_dir=str(root_path),
                split_file=str(split_file),
                transform=transform
            )
        else:
            # Fallback: scan faces directory
            super().__init__(
                root_dir=str(faces_path),
                transform=transform
            )


class FaceForensicsDataset(DeepfakeDataset):
    """Dataset for FaceForensics++."""
    
    MANIPULATION_METHODS = [
        "Deepfakes",
        "Face2Face", 
        "FaceSwap",
        "NeuralTextures"
    ]
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        faces_dir: Optional[str] = None,
        methods: Optional[List[str]] = None,
        compression: str = "c23",
    ):
        """
        Initialize FaceForensics++ dataset.
        
        Args:
            root_dir: Root directory of datasets
            split: One of 'train', 'val', 'test'
            transform: Albumentations transform
            faces_dir: Directory with extracted faces
            methods: List of manipulation methods to include (default: all)
            compression: Compression level ('raw', 'c23', 'c40')
        """
        self.split = split
        self.methods = methods or self.MANIPULATION_METHODS
        self.compression = compression
        
        root_path = Path(root_dir)
        
        # Split file path
        split_file = root_path / "splits" / f"ff_{split}.txt"
        
        if split_file.exists():
            super().__init__(
                root_dir=str(root_path),
                split_file=str(split_file),
                transform=transform
            )
        else:
            # Fallback: scan faces directory
            faces_path = faces_dir or (root_path / "faces" / "FaceForensics")
            super().__init__(
                root_dir=str(faces_path),
                transform=transform
            )


class CombinedDataset(Dataset):
    """
    Combined dataset from multiple sources with balanced sampling.
    """
    
    def __init__(
        self,
        datasets: List[Dataset],
        balance_classes: bool = True,
    ):
        """
        Initialize combined dataset.
        
        Args:
            datasets: List of datasets to combine
            balance_classes: Whether to balance classes during sampling
        """
        self.datasets = datasets
        self.balance_classes = balance_classes
        
        # Combine all samples
        self.samples = []
        self.dataset_indices = []  # Track which dataset each sample came from
        
        for dataset_idx, dataset in enumerate(datasets):
            if hasattr(dataset, 'samples'):
                for sample in dataset.samples:
                    self.samples.append(sample)
                    self.dataset_indices.append(dataset_idx)
        
        # Store transforms from first dataset
        self.transform = datasets[0].transform if datasets else None
        
        print(f"Combined dataset: {len(self.samples)} total samples")
        self._print_stats()
    
    def _print_stats(self):
        """Print combined statistics."""
        labels = [s[1] for s in self.samples]
        real_count = labels.count(0)
        fake_count = labels.count(1)
        print(f"  Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image, label
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling."""
        labels = [s[1] for s in self.samples]
        class_counts = [max(1, labels.count(0)), max(1, labels.count(1))]
        weights = [1.0 / class_counts[label] for label in labels]
        return torch.DoubleTensor(weights)


def create_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    use_combined: bool = True,
    balance_training: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        root_dir: Root directory containing datasets
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        use_combined: Whether to combine FF++ and Celeb-DF
        balance_training: Whether to use balanced sampling for training
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .transforms import get_train_transforms, get_val_transforms
    
    train_transform = train_transform or get_train_transforms()
    val_transform = val_transform or get_val_transforms()
    
    root_path = Path(root_dir)
    
    # Create datasets
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    # Celeb-DF
    celebdf_splits = root_path / "splits"
    if (celebdf_splits / "celebdf_train.txt").exists():
        train_datasets.append(CelebDFDataset(root_dir, "train", train_transform))
        val_datasets.append(CelebDFDataset(root_dir, "val", val_transform))
        test_datasets.append(CelebDFDataset(root_dir, "test", val_transform))
    
    # FaceForensics++
    if (celebdf_splits / "ff_train.txt").exists():
        train_datasets.append(FaceForensicsDataset(root_dir, "train", train_transform))
        val_datasets.append(FaceForensicsDataset(root_dir, "val", val_transform))
        test_datasets.append(FaceForensicsDataset(root_dir, "test", val_transform))
    
    # Combine or use single dataset
    if use_combined and len(train_datasets) > 1:
        train_dataset = CombinedDataset(train_datasets)
        train_dataset.transform = train_transform
        val_dataset = CombinedDataset(val_datasets)
        val_dataset.transform = val_transform
        test_dataset = CombinedDataset(test_datasets)
        test_dataset.transform = val_transform
    elif train_datasets:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]
        test_dataset = test_datasets[0]
    else:
        # Fallback: scan directories
        print("No split files found, scanning directories...")
        faces_dir = root_path / "faces"
        train_dataset = DeepfakeDataset(str(faces_dir), transform=train_transform)
        val_dataset = train_dataset  # Use same for all
        test_dataset = train_dataset
    
    # Create samplers
    if balance_training and hasattr(train_dataset, 'get_sample_weights'):
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
