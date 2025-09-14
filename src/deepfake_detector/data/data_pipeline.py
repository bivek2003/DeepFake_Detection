"""
PyTorch Data Pipeline for Deepfake Detection

Integrates video and audio processing with PyTorch datasets and dataloaders.
Handles train/val/test splitting with proper stratification.

Author: Your Name
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Callable
import json
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import pickle
from tqdm import tqdm
import cv2
from dataclasses import dataclass

from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Information about data splits"""
    train_paths: List[str]
    train_labels: List[int]
    val_paths: List[str]
    val_labels: List[int]
    test_paths: List[str]
    test_labels: List[int]
    split_info: Dict[str, any]


class DeepfakeVideoDataset(Dataset):
    """PyTorch Dataset for video-based deepfake detection"""
    
    def __init__(self,
                 video_paths: List[str],
                 labels: List[int],
                 video_processor: Optional[VideoProcessor] = None,
                 transform: Optional[transforms.Compose] = None,
                 max_faces_per_video: int = 1,
                 preload: bool = False):
        """
        Initialize video dataset
        
        Args:
            video_paths: List of video file paths
            labels: List of labels (0=real, 1=fake)
            video_processor: VideoProcessor instance
            transform: Torchvision transforms
            max_faces_per_video: Maximum faces to extract per video
            preload: Whether to preload all data into memory
        """
        self.video_paths = video_paths
        self.labels = labels
        self.video_processor = video_processor or VideoProcessor()
        self.transform = transform
        self.max_faces_per_video = max_faces_per_video
        self.preload = preload
        
        # Preload data if requested
        if self.preload:
            logger.info("Preloading video data...")
            self.preloaded_data = self._preload_all_data()
        else:
            self.preloaded_data = None
        
        logger.info(f"Initialized VideoDataset with {len(video_paths)} videos")
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        if self.preloaded_data is not None:
            # Use preloaded data
            face_crop = self.preloaded_data[idx]
        else:
            # Load data on demand
            video_path = self.video_paths[idx]
            face_crops = self.video_processor.extract_faces_from_video(video_path)
            
            if face_crops:
                # Take first face or random face if multiple
                face_crop = face_crops[0]
            else:
                # Return black image if no faces found
                face_crop = np.zeros((224, 224, 3), dtype=np.uint8)
        
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            # Convert numpy to PIL Image for torchvision transforms
            from PIL import Image
            if isinstance(face_crop, np.ndarray):
                face_crop = Image.fromarray(face_crop)
            face_crop = self.transform(face_crop)
        else:
            # Convert to tensor manually
            face_crop = torch.from_numpy(face_crop).permute(2, 0, 1).float() / 255.0
        
        return face_crop, torch.tensor(label, dtype=torch.long)
    
    def _preload_all_data(self) -> List[np.ndarray]:
        """Preload all video data for faster training"""
        preloaded = []
        for video_path in tqdm(self.video_paths, desc="Preloading videos"):
            try:
                face_crops = self.video_processor.extract_faces_from_video(video_path)
                if face_crops:
                    preloaded.append(face_crops[0])
                else:
                    preloaded.append(np.zeros((224, 224, 3), dtype=np.uint8))
            except Exception as e:
                logger.warning(f"Error loading {video_path}: {e}")
                preloaded.append(np.zeros((224, 224, 3), dtype=np.uint8))
        return preloaded


class DeepfakeAudioDataset(Dataset):
    """PyTorch Dataset for audio-based deepfake detection"""
    
    def __init__(self,
                 audio_paths: List[str], 
                 labels: List[int],
                 audio_processor: Optional[AudioProcessor] = None,
                 transform: Optional[Callable] = None,
                 feature_type: str = "mfcc",
                 preload: bool = False):
        """
        Initialize audio dataset
        
        Args:
            audio_paths: List of audio file paths
            labels: List of labels (0=real, 1=fake) 
            audio_processor: AudioProcessor instance
            transform: Transform function for audio features
            feature_type: Type of features to use ('mfcc', 'mel', 'raw')
            preload: Whether to preload all data into memory
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.audio_processor = audio_processor or AudioProcessor()
        self.transform = transform
        self.feature_type = feature_type
        self.preload = preload
        
        # Preload data if requested
        if self.preload:
            logger.info("Preloading audio data...")
            self.preloaded_data = self._preload_all_data()
        else:
            self.preloaded_data = None
            
        logger.info(f"Initialized AudioDataset with {len(audio_paths)} audio files")
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        if self.preloaded_data is not None:
            # Use preloaded data
            features = self.preloaded_data[idx]
        else:
            # Load data on demand
            audio_path = self.audio_paths[idx]
            try:
                audio, sr = self.audio_processor.load_audio(audio_path)
                extracted_features = self.audio_processor.extract_features(audio, sr)
                
                # Select feature type
                if self.feature_type == "mfcc":
                    features = extracted_features.mfcc
                elif self.feature_type == "mel":
                    features = extracted_features.mel_spectrogram
                elif self.feature_type == "raw":
                    features = audio.reshape(1, -1)  # Add channel dimension
                else:
                    raise ValueError(f"Unknown feature type: {self.feature_type}")
                    
            except Exception as e:
                logger.warning(f"Error loading {audio_path}: {e}")
                # Return zeros if loading fails
                if self.feature_type == "mfcc":
                    features = np.zeros((13, 94))  # Typical MFCC shape
                elif self.feature_type == "mel":
                    features = np.zeros((128, 94))  # Typical mel shape
                elif self.feature_type == "raw":
                    features = np.zeros((1, 48000))  # 3 seconds at 16kHz
        
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            features = self.transform(features)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return features_tensor, label_tensor
    
    def _preload_all_data(self) -> List[np.ndarray]:
        """Preload all audio data for faster training"""
        preloaded = []
        for audio_path in tqdm(self.audio_paths, desc="Preloading audio"):
            try:
                audio, sr = self.audio_processor.load_audio(audio_path)
                extracted_features = self.audio_processor.extract_features(audio, sr)
                
                if self.feature_type == "mfcc":
                    features = extracted_features.mfcc
                elif self.feature_type == "mel":
                    features = extracted_features.mel_spectrogram
                elif self.feature_type == "raw":
                    features = audio.reshape(1, -1)
                
                preloaded.append(features)
                
            except Exception as e:
                logger.warning(f"Error preloading {audio_path}: {e}")
                # Add zero features for failed loads
                if self.feature_type == "mfcc":
                    features = np.zeros((13, 94))
                elif self.feature_type == "mel":
                    features = np.zeros((128, 94))
                elif self.feature_type == "raw":
                    features = np.zeros((1, 48000))
                preloaded.append(features)
                
        return preloaded


class DataSplitter:
    """Handles stratified train/validation/test splitting"""
    
    def __init__(self, 
                 test_size: float = 0.2,
                 val_size: float = 0.1, 
                 random_state: int = 42,
                 stratify: bool = True):
        """
        Initialize data splitter
        
        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of data for validation set
            random_state: Random seed for reproducibility
            stratify: Whether to maintain class balance in splits
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify
    
    def split_data(self, file_paths: List[str], labels: List[int]) -> DataSplit:
        """
        Split data into train/validation/test sets
        
        Args:
            file_paths: List of file paths
            labels: List of corresponding labels
            
        Returns:
            DataSplit object with all splits and metadata
        """
        if len(file_paths) != len(labels):
            raise ValueError("Number of file paths must match number of labels")
        
        # Convert to numpy for easier handling
        paths_array = np.array(file_paths)
        labels_array = np.array(labels)
        
        if self.stratify:
            # Stratified splitting to maintain class balance
            
            # First split: separate test set
            sss_test = StratifiedShuffleSplit(
                n_splits=1, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            train_val_idx, test_idx = next(sss_test.split(paths_array, labels_array))
            
            # Second split: separate train and validation
            val_size_adjusted = self.val_size / (1 - self.test_size)
            sss_val = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_size_adjusted,
                random_state=self.random_state
            )
            train_idx, val_idx = next(sss_val.split(
                paths_array[train_val_idx], 
                labels_array[train_val_idx]
            ))
            
            # Convert back to original indices
            train_idx = train_val_idx[train_idx]
            val_idx = train_val_idx[val_idx]
            
        else:
            # Random splitting
            train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
                file_paths, labels,
                test_size=self.test_size,
                random_state=self.random_state
            )
            
            val_size_adjusted = self.val_size / (1 - self.test_size)
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_val_paths, train_val_labels,
                test_size=val_size_adjusted,
                random_state=self.random_state
            )
            
            # Convert to indices for consistency
            train_idx = [i for i, p in enumerate(file_paths) if p in train_paths]
            val_idx = [i for i, p in enumerate(file_paths) if p in val_paths] 
            test_idx = [i for i, p in enumerate(file_paths) if p in test_paths]
        
        # Extract splits
        train_paths = paths_array[train_idx].tolist()
        train_labels = labels_array[train_idx].tolist()
        val_paths = paths_array[val_idx].tolist()
        val_labels = labels_array[val_idx].tolist()
        test_paths = paths_array[test_idx].tolist()
        test_labels = labels_array[test_idx].tolist()
        
        # Calculate split statistics
        total_samples = len(file_paths)
        split_info = {
            "total_samples": total_samples,
            "train_samples": len(train_paths),
            "val_samples": len(val_paths),
            "test_samples": len(test_paths),
            "train_ratio": len(train_paths) / total_samples,
            "val_ratio": len(val_paths) / total_samples,
            "test_ratio": len(test_paths) / total_samples,
            "class_distribution": self._calculate_class_distribution(
                train_labels, val_labels, test_labels
            ),
            "stratified": self.stratify,
            "random_state": self.random_state
        }
        
        return DataSplit(
            train_paths=train_paths,
            train_labels=train_labels,
            val_paths=val_paths,
            val_labels=val_labels,
            test_paths=test_paths,
            test_labels=test_labels,
            split_info=split_info
        )
    
    def _calculate_class_distribution(self, train_labels, val_labels, test_labels):
        """Calculate class distribution across splits"""
        def get_class_counts(labels):
            unique, counts = np.unique(labels, return_counts=True)
            return dict(zip(unique.astype(int), counts.astype(int)))
        
        return {
            "train": get_class_counts(train_labels),
            "val": get_class_counts(val_labels),
            "test": get_class_counts(test_labels)
        }
    
    def save_split(self, data_split: DataSplit, save_path: Union[str, Path]):
        """Save data split to disk"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        split_data = {
            "train_paths": data_split.train_paths,
            "train_labels": data_split.train_labels,
            "val_paths": data_split.val_paths,
            "val_labels": data_split.val_labels,
            "test_paths": data_split.test_paths,
            "test_labels": data_split.test_labels,
            "split_info": data_split.split_info
        }
        
        with open(save_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        logger.info(f"Saved data split to {save_path}")
    
    def load_split(self, load_path: Union[str, Path]) -> DataSplit:
        """Load data split from disk"""
        with open(load_path, 'r') as f:
            split_data = json.load(f)
        
        return DataSplit(
            train_paths=split_data["train_paths"],
            train_labels=split_data["train_labels"],
            val_paths=split_data["val_paths"],
            val_labels=split_data["val_labels"],
            test_paths=split_data["test_paths"],
            test_labels=split_data["test_labels"],
            split_info=split_data["split_info"]
        )
    
    def print_split_info(self, data_split: DataSplit):
        """Print detailed information about data splits"""
        info = data_split.split_info
        
        print("📊 Data Split Information")
        print("=" * 40)
        print(f"Total samples: {info['total_samples']:,}")
        print(f"Stratified: {'Yes' if info['stratified'] else 'No'}")
        print(f"Random seed: {info['random_state']}")
        print()
        
        # Split sizes
        print("📈 Split Sizes:")
        print(f"  Train: {info['train_samples']:,} ({info['train_ratio']:.1%})")
        print(f"  Val:   {info['val_samples']:,} ({info['val_ratio']:.1%})")
        print(f"  Test:  {info['test_samples']:,} ({info['test_ratio']:.1%})")
        print()
        
        # Class distribution
        print("🎯 Class Distribution:")
        for split_name, class_counts in info['class_distribution'].items():
            total = sum(class_counts.values())
            print(f"  {split_name.upper()}:")
            for class_id, count in class_counts.items():
                class_name = "Real" if class_id == 0 else "Fake"
                percentage = count / total * 100 if total > 0 else 0
                print(f"    {class_name}: {count:,} ({percentage:.1f}%)")
        print()


class VideoAugmentations:
    """Video-specific augmentations for deepfake detection"""
    
    @staticmethod
    def get_train_transforms(image_size: int = 224) -> transforms.Compose:
        """Get training transforms with augmentations"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
        ])
    
    @staticmethod
    def get_val_transforms(image_size: int = 224) -> transforms.Compose:
        """Get validation transforms without augmentations"""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class AudioAugmentations:
    """Audio-specific augmentations for deepfake detection"""
    
    @staticmethod
    def add_noise(features: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """Add Gaussian noise to features"""
        noise = np.random.normal(0, noise_factor, features.shape)
        return features + noise
    
    @staticmethod
    def frequency_mask(spectrogram: np.ndarray, freq_mask_param: int = 5) -> np.ndarray:
        """Apply frequency masking to spectrogram"""
        cloned = spectrogram.copy()
        num_freq_bins = cloned.shape[0]
        
        f = np.random.randint(0, freq_mask_param)
        f_0 = np.random.randint(0, num_freq_bins - f)
        
        cloned[f_0:f_0+f, :] = 0
        return cloned
    
    @staticmethod
    def time_mask(spectrogram: np.ndarray, time_mask_param: int = 10) -> np.ndarray:
        """Apply time masking to spectrogram"""
        cloned = spectrogram.copy()
        num_time_bins = cloned.shape[1]
        
        t = np.random.randint(0, time_mask_param)
        t_0 = np.random.randint(0, num_time_bins - t)
        
        cloned[:, t_0:t_0+t] = 0
        return cloned


class DataPipelineManager:
    """Main manager for data pipeline creation and management"""
    
    def __init__(self, 
                 data_root: str = "./datasets",
                 processed_root: str = "./preprocessed"):
        self.data_root = Path(data_root)
        self.processed_root = Path(processed_root)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        self.data_splitter = DataSplitter()
        
    def create_video_dataloaders(self,
                                video_paths: List[str],
                                labels: List[int],
                                batch_size: int = 32,
                                num_workers: int = 4,
                                split_data: bool = True,
                                preload: bool = False) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for video data
        
        Args:
            video_paths: List of video file paths
            labels: List of labels
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            split_data: Whether to split into train/val/test
            preload: Whether to preload all data into memory
            
        Returns:
            Dictionary with train/val/test DataLoaders
        """
        if split_data:
            # Split the data
            data_split = self.data_splitter.split_data(video_paths, labels)
            
            # Save split information
            split_path = self.processed_root / "video_split.json"
            self.data_splitter.save_split(data_split, split_path)
            self.data_splitter.print_split_info(data_split)
            
            # Create transforms
            train_transforms = VideoAugmentations.get_train_transforms()
            val_transforms = VideoAugmentations.get_val_transforms()
            
            # Create datasets
            train_dataset = DeepfakeVideoDataset(
                data_split.train_paths,
                data_split.train_labels,
                self.video_processor,
                train_transforms,
                preload=preload
            )
            
            val_dataset = DeepfakeVideoDataset(
                data_split.val_paths,
                data_split.val_labels,
                self.video_processor,
                val_transforms,
                preload=preload
            )
            
            test_dataset = DeepfakeVideoDataset(
                data_split.test_paths,
                data_split.test_labels,
                self.video_processor,
                val_transforms,
                preload=preload
            )
            
            # Create dataloaders
            dataloaders = {
                "train": DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                ),
                "test": DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
            }
            
        else:
            # Create single dataloader without splitting
            transforms_compose = VideoAugmentations.get_val_transforms()
            dataset = DeepfakeVideoDataset(
                video_paths,
                labels,
                self.video_processor,
                transforms_compose,
                preload=preload
            )
            
            dataloaders = {
                "full": DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
            }
        
        return dataloaders
    
    def create_audio_dataloaders(self,
                               audio_paths: List[str],
                               labels: List[int],
                               batch_size: int = 32,
                               num_workers: int = 4,
                               feature_type: str = "mfcc",
                               split_data: bool = True,
                               preload: bool = False) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for audio data
        
        Args:
            audio_paths: List of audio file paths
            labels: List of labels
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes
            feature_type: Type of features ('mfcc', 'mel', 'raw')
            split_data: Whether to split into train/val/test
            preload: Whether to preload all data into memory
            
        Returns:
            Dictionary with train/val/test DataLoaders
        """
        if split_data:
            # Split the data
            data_split = self.data_splitter.split_data(audio_paths, labels)
            
            # Save split information
            split_path = self.processed_root / "audio_split.json"
            self.data_splitter.save_split(data_split, split_path)
            self.data_splitter.print_split_info(data_split)
            
            # Create datasets
            train_dataset = DeepfakeAudioDataset(
                data_split.train_paths,
                data_split.train_labels,
                self.audio_processor,
                feature_type=feature_type,
                preload=preload
            )
            
            val_dataset = DeepfakeAudioDataset(
                data_split.val_paths,
                data_split.val_labels,
                self.audio_processor,
                feature_type=feature_type,
                preload=preload
            )
            
            test_dataset = DeepfakeAudioDataset(
                data_split.test_paths,
                data_split.test_labels,
                self.audio_processor,
                feature_type=feature_type,
                preload=preload
            )
            
            # Create dataloaders
            dataloaders = {
                "train": DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                ),
                "test": DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
            }
            
        else:
            # Create single dataloader without splitting
            dataset = DeepfakeAudioDataset(
                audio_paths,
                labels,
                self.audio_processor,
                feature_type=feature_type,
                preload=preload
            )
            
            dataloaders = {
                "full": DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
            }
        
        return dataloaders
    
    def create_sample_data_pipeline(self, 
                                  num_videos: int = 20,
                                  num_audios: int = 20) -> Dict[str, DataLoader]:
        """
        Create sample data pipeline for testing
        
        Args:
            num_videos: Number of sample videos
            num_audios: Number of sample audio files
            
        Returns:
            Dictionary with sample dataloaders
        """
        # Create sample paths and labels
        video_paths = []
        video_labels = []
        audio_paths = []
        audio_labels = []
        
        # Video data
        for i in range(num_videos // 2):
            video_paths.append(f"sample_real_video_{i}.mp4")
            video_labels.append(0)  # Real
            video_paths.append(f"sample_fake_video_{i}.mp4") 
            video_labels.append(1)  # Fake
        
        # Audio data  
        for i in range(num_audios // 2):
            audio_paths.append(f"sample_real_audio_{i}.wav")
            audio_labels.append(0)  # Real
            audio_paths.append(f"sample_fake_audio_{i}.wav")
            audio_labels.append(1)  # Fake
        
        # Note: These would be actual file paths in real usage
        logger.warning("Sample data pipeline created with placeholder paths")
        
        return {
            "video_paths": video_paths,
            "video_labels": video_labels,
            "audio_paths": audio_paths,
            "audio_labels": audio_labels
        }


def main():
    """Demonstration of data pipeline functionality"""
    # Initialize pipeline manager
    pipeline = DataPipelineManager()
    
    print("🔗 PyTorch Data Pipeline Demo")
    print("=" * 50)
    
    # Create sample data
    sample_data = pipeline.create_sample_data_pipeline()
    
    print("\nCapabilities:")
    print("• PyTorch Dataset integration for video and audio")
    print("• Stratified train/val/test splitting")
    print("• Data augmentation pipelines")  
    print("• Efficient DataLoaders with multiprocessing")
    print("• Memory preloading options")
    print("• Automatic split saving/loading")
    print("• Class balance preservation")
    
    print(f"\nDataset Features:")
    print(f"• Video: Face extraction → 224x224 RGB")
    print(f"• Audio: MFCC/Mel/Raw features")
    print(f"• Labels: 0=Real, 1=Fake")
    print(f"• Transforms: ImageNet normalization")
    
    print(f"\nSample Data Created:")
    print(f"• Videos: {len(sample_data['video_paths'])}")
    print(f"• Audio files: {len(sample_data['audio_paths'])}")
    print(f"• Class balance: 50% Real, 50% Fake")
    
    # Example usage
    print(f"\nExample Usage:")
    print(f"```python")
    print(f"# Create video dataloaders")
    print(f"video_loaders = pipeline.create_video_dataloaders(")
    print(f"    video_paths, labels, batch_size=32")
    print(f")")
    print(f"")
    print(f"# Create audio dataloaders")
    print(f"audio_loaders = pipeline.create_audio_dataloaders(")
    print(f"    audio_paths, labels, feature_type='mfcc'")
    print(f")")
    print(f"")
    print(f"# Training loop")
    print(f"for batch_idx, (data, labels) in enumerate(video_loaders['train']):")
    print(f"    # data.shape: [batch_size, 3, 224, 224]")
    print(f"    # labels.shape: [batch_size]")
    print(f"    pass")
    print(f"```")


if __name__ == "__main__":
    main()
