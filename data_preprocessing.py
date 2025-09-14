"""
Phase 1: Data Acquisition & Preprocessing for Deepfake Detection App
This module handles dataset acquisition, preprocessing, and preparation for training.
Supports both image/video and audio deepfake datasets.
"""

import os
import cv2
import numpy as np
import pandas as pd
import librosa
import torchaudio
from pathlib import Path
import json
import urllib.request
import zipfile
import tarfile
from typing import List, Tuple, Dict, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import requests
import warnings
warnings.filterwarnings('ignore')

class DatasetManager:
    """Manages dataset downloads, verification, and organization"""
    
    def __init__(self, data_root: str = "./datasets"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        
        # Dataset information from roadmap
        self.datasets_info = {
            "faceforensics": {
                "name": "FaceForensics++",
                "type": "video",
                "url": "https://github.com/ondyari/FaceForensics",  # Requires form submission
                "description": "1,000 videos, 1.8M manipulated images",
                "requires_form": True
            },
            "dfdc": {
                "name": "DFDC",
                "type": "video", 
                "url": "https://ai.facebook.com/datasets/dfdc/",  # Facebook release
                "description": "100K+ clips, 3,426 actors",
                "requires_form": True
            },
            "celebdf": {
                "name": "Celeb-DF",
                "type": "video",
                "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
                "description": "590 originals + 5,639 deepfakes",
                "requires_form": True
            },
            "wilddeepfake": {
                "name": "WildDeepfake", 
                "type": "video",
                "url": "https://huggingface.co/datasets/wilddeepfake",
                "description": "707 real 'in-the-wild' deepfake videos"
            },
            "asvspoof2021": {
                "name": "ASVspoof 2021",
                "type": "audio",
                "url": "https://zenodo.org/record/4837263",
                "description": "TTS/VC speech deepfake dataset"
            },
            "fakeavceleb": {
                "name": "FakeAVCeleb",
                "type": "multimodal",
                "url": "https://openreview.net/forum?id=TAXFsg6ZaOl",
                "description": "Synchronized fake video + lip-synced fake audio"
            }
        }
    
    def print_dataset_info(self):
        """Print information about available datasets"""
        print("Available Deepfake Detection Datasets:")
        print("=" * 50)
        for key, info in self.datasets_info.items():
            print(f"Dataset: {info['name']}")
            print(f"Type: {info['type']}")
            print(f"Description: {info['description']}")
            print(f"URL: {info['url']}")
            if info.get('requires_form'):
                print("⚠️  Requires form submission for access")
            print("-" * 30)
    
    def download_sample_data(self):
        """Download sample deepfake data for demonstration"""
        print("Creating sample dataset structure...")
        
        # Create directory structure
        sample_dir = self.data_root / "sample_data"
        sample_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (sample_dir / "real" / "videos").mkdir(parents=True, exist_ok=True)
        (sample_dir / "fake" / "videos").mkdir(parents=True, exist_ok=True)
        (sample_dir / "real" / "audio").mkdir(parents=True, exist_ok=True)
        (sample_dir / "fake" / "audio").mkdir(parents=True, exist_ok=True)
        
        print(f"Sample dataset structure created at: {sample_dir}")
        return sample_dir

class VideoPreprocessor:
    """Handles video preprocessing for deepfake detection"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def extract_faces_from_video(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """Extract face regions from video frames"""
        cap = cv2.VideoCapture(video_path)
        faces = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Extract largest face
            if len(detected_faces) > 0:
                # Get the largest face
                largest_face = max(detected_faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract and resize face
                face = frame_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face, self.target_size)
                faces.append(face_resized)
            
            frame_count += 1
        
        cap.release()
        return faces
    
    def create_video_augmentations(self) -> transforms.Compose:
        """Create augmentation pipeline for video frames"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class AudioPreprocessor:
    """Handles audio preprocessing for deepfake detection"""
    
    def __init__(self, sample_rate: int = 16000, duration: float = 3.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            # Load audio using librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or trim to fixed length
            if len(audio) > self.n_samples:
                audio = audio[:self.n_samples]
            else:
                audio = np.pad(audio, (0, max(0, self.n_samples - len(audio))), 'constant')
            
            return audio
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return np.zeros(self.n_samples)
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract various audio features for deepfake detection"""
        features = {}
        
        # MFCC features
        features['mfcc'] = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
        
        # Mel spectrogram
        features['mel_spectrogram'] = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        
        return features
    
    def create_audio_augmentations(self, audio: np.ndarray) -> List[np.ndarray]:
        """Create augmented versions of audio"""
        augmented = []
        
        # Original
        augmented.append(audio)
        
        # Add noise
        noise_factor = 0.005
        noise = np.random.randn(len(audio))
        augmented.append(audio + noise_factor * noise)
        
        # Time shift
        shift_max = int(0.2 * len(audio))
        shift = np.random.randint(-shift_max, shift_max)
        augmented.append(np.roll(audio, shift))
        
        # Speed change (pitch shift)
        try:
            augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
            augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
        except:
            pass
        
        return augmented

class DeepfakeDataset(Dataset):
    """PyTorch Dataset for deepfake detection"""
    
    def __init__(self, data_paths: List[str], labels: List[int], 
                 modality: str = "video", transform=None, preprocessor=None):
        self.data_paths = data_paths
        self.labels = labels
        self.modality = modality
        self.transform = transform
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        path = self.data_paths[idx]
        label = self.labels[idx]
        
        if self.modality == "video":
            # For video, extract faces and return first face
            if self.preprocessor:
                faces = self.preprocessor.extract_faces_from_video(path, max_frames=1)
                if faces:
                    data = faces[0]
                else:
                    # Return black image if no face found
                    data = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                # Simple video frame extraction
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                if ret:
                    data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    data = cv2.resize(data, (224, 224))
                else:
                    data = np.zeros((224, 224, 3), dtype=np.uint8)
                cap.release()
                
        elif self.modality == "audio":
            if self.preprocessor:
                data = self.preprocessor.load_audio(path)
            else:
                # Simple audio loading
                try:
                    data, _ = librosa.load(path, sr=16000, duration=3.0)
                except:
                    data = np.zeros(48000)  # 3 seconds at 16kHz
        
        if self.transform:
            data = self.transform(data)
            
        return data, label

class DataSplitter:
    """Handles train/validation/test splitting with proper balance"""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split_data(self, data_paths: List[str], labels: List[int]) -> Dict[str, Tuple[List[str], List[int]]]:
        """Split data into train/val/test with balanced classes"""
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            data_paths, labels, 
            test_size=self.test_size, 
            stratify=labels,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_size_adjusted,
            stratify=y_trainval,
            random_state=self.random_state
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def print_split_info(self, splits: Dict[str, Tuple[List[str], List[int]]]):
        """Print information about data splits"""
        print("Data Split Information:")
        print("=" * 30)
        
        for split_name, (paths, labels) in splits.items():
            total = len(labels)
            real_count = sum(1 for l in labels if l == 0)
            fake_count = sum(1 for l in labels if l == 1)
            
            print(f"{split_name.upper()} SET:")
            print(f"  Total samples: {total}")
            print(f"  Real samples: {real_count} ({real_count/total*100:.1f}%)")
            print(f"  Fake samples: {fake_count} ({fake_count/total*100:.1f}%)")
            print()

class Phase1Manager:
    """Main manager for Phase 1 implementation"""
    
    def __init__(self, data_root: str = "./datasets"):
        self.dataset_manager = DatasetManager(data_root)
        self.video_preprocessor = VideoPreprocessor()
        self.audio_preprocessor = AudioPreprocessor()
        self.data_splitter = DataSplitter()
        
    def setup_datasets(self):
        """Setup and prepare datasets"""
        print("=== PHASE 1: DATA ACQUISITION & PREPROCESSING ===\n")
        
        # Print dataset information
        self.dataset_manager.print_dataset_info()
        
        # Create sample dataset structure
        sample_dir = self.dataset_manager.download_sample_data()
        
        return sample_dir
    
    def create_sample_data(self, sample_dir: Path):
        """Create sample data for demonstration"""
        print("\nCreating sample data for demonstration...")
        
        # Create dummy video files (just file paths for demo)
        video_paths = []
        video_labels = []
        
        # Real videos
        for i in range(10):
            video_path = sample_dir / "real" / "videos" / f"real_video_{i:03d}.mp4"
            video_paths.append(str(video_path))
            video_labels.append(0)  # 0 = real
            
        # Fake videos  
        for i in range(10):
            video_path = sample_dir / "fake" / "videos" / f"fake_video_{i:03d}.mp4"
            video_paths.append(str(video_path))
            video_labels.append(1)  # 1 = fake
            
        # Audio files
        audio_paths = []
        audio_labels = []
        
        # Real audio
        for i in range(10):
            audio_path = sample_dir / "real" / "audio" / f"real_audio_{i:03d}.wav"
            audio_paths.append(str(audio_path))
            audio_labels.append(0)
            
        # Fake audio
        for i in range(10):
            audio_path = sample_dir / "fake" / "audio" / f"fake_audio_{i:03d}.wav"
            audio_paths.append(str(audio_path))
            audio_labels.append(1)
        
        return {
            'video': (video_paths, video_labels),
            'audio': (audio_paths, audio_labels)
        }
    
    def demonstrate_preprocessing(self):
        """Demonstrate preprocessing capabilities"""
        print("\n=== PREPROCESSING DEMONSTRATION ===")
        
        # Video preprocessing demo
        print("\n1. Video Preprocessing Pipeline:")
        print("   - Face detection using Haar cascades")
        print("   - Face extraction and resizing to 224x224")
        print("   - Data augmentation: flip, color jitter, rotation")
        print("   - Normalization using ImageNet statistics")
        
        # Show augmentation pipeline
        video_transforms = self.video_preprocessor.create_video_augmentations()
        print(f"   - Transforms: {video_transforms}")
        
        # Audio preprocessing demo
        print("\n2. Audio Preprocessing Pipeline:")
        print("   - Resampling to 16kHz")
        print("   - Fixed duration: 3.0 seconds")
        print("   - Feature extraction: MFCC, spectral features, mel-spectrogram")
        print("   - Augmentations: noise addition, time shifting, speed change")
        
        # Create sample datasets
        sample_dir = self.setup_datasets()
        sample_data = self.create_sample_data(sample_dir)
        
        # Demonstrate data splitting
        print("\n3. Data Splitting:")
        for modality, (paths, labels) in sample_data.items():
            print(f"\n{modality.upper()} DATA:")
            splits = self.data_splitter.split_data(paths, labels)
            self.data_splitter.print_split_info(splits)
        
        print("\n=== PHASE 1 SETUP COMPLETE ===")
        print("\nNext steps:")
        print("1. Download actual datasets using provided links")
        print("2. Update paths in the code to point to real data")
        print("3. Run preprocessing on actual video/audio files")
        print("4. Proceed to Phase 2: Model Development")
        
        return sample_dir

def main():
    """Main function to run Phase 1"""
    # Initialize Phase 1 manager
    phase1 = Phase1Manager()
    
    # Run demonstration
    sample_dir = phase1.demonstrate_preprocessing()
    
    # Example of creating data loaders
    print("\n=== DATA LOADER EXAMPLE ===")
    
    # Create sample video dataset
    sample_data = phase1.create_sample_data(sample_dir)
    video_paths, video_labels = sample_data['video']
    
    # Split the data
    splits = phase1.data_splitter.split_data(video_paths, video_labels)
    
    # Create datasets
    train_paths, train_labels = splits['train']
    val_paths, val_labels = splits['val']
    
    # Video transforms
    video_transforms = phase1.video_preprocessor.create_video_augmentations()
    
    # Create PyTorch datasets (demo - won't work without actual video files)
    print("Creating PyTorch datasets...")
    print(f"Training set: {len(train_paths)} samples")
    print(f"Validation set: {len(val_paths)} samples")
    
    # train_dataset = DeepfakeDataset(
    #     train_paths, train_labels, 
    #     modality="video", 
    #     transform=video_transforms,
    #     preprocessor=phase1.video_preprocessor
    # )
    
    # train_loader = DataLoader(
    #     train_dataset, 
    #     batch_size=32, 
    #     shuffle=True, 
    #     num_workers=4
    # )
    
    print("\nPhase 1 implementation ready!")
    print("Replace sample data with actual dataset files to begin training.")

if __name__ == "__main__":
    main()
