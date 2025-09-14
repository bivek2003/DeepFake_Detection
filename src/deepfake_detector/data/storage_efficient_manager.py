"""
Storage-Efficient Dataset Manager for Deepfake Detection

Optimized for limited storage (260GB) while maintaining professional functionality.
Focuses on smaller datasets and synthetic data generation.

Author: Bivek Sharma Panthi
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json
import requests
from tqdm import tqdm
import subprocess
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StorageEfficientDatasetManager:
    """Dataset manager optimized for limited storage"""
    
    def __init__(self, data_root: str = "./datasets", max_storage_gb: float = 50.0):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.max_storage_gb = max_storage_gb
        
        # Storage-friendly dataset selection
        self.recommended_datasets = {
            "wilddeepfake": {
                "name": "WildDeepfake",
                "size_gb": 4.2,
                "priority": 1,
                "url": "https://huggingface.co/datasets/wilddeepfake",
                "description": "707 'in-the-wild' deepfake videos",
                "download_method": "huggingface"
            },
            "celebdf_sample": {
                "name": "Celeb-DF (Sample)",
                "size_gb": 5.0,
                "priority": 1,
                "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
                "description": "Subset of Celeb-DF dataset (1000 videos)",
                "download_method": "manual_subset"
            },
            "asvspoof_eval": {
                "name": "ASVspoof 2021 (Eval Only)",
                "size_gb": 5.0,
                "priority": 1,
                "url": "https://zenodo.org/record/4837263",
                "description": "Evaluation set only for audio deepfakes",
                "download_method": "zenodo_subset"
            },
            "synthetic_data": {
                "name": "Generated Synthetic Dataset",
                "size_gb": 10.0,
                "priority": 1,
                "url": "local_generation",
                "description": "AI-generated faces and synthetic audio",
                "download_method": "generate"
            }
        }
        
        logger.info(f"Storage-efficient manager initialized (max {max_storage_gb}GB)")
    
    def check_storage_usage(self) -> Dict[str, float]:
        """Check current storage usage"""
        total_usage = 0.0
        dataset_usage = {}
        
        if self.data_root.exists():
            for item in self.data_root.iterdir():
                if item.is_dir():
                    size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    size_gb = size_bytes / (1024**3)
                    dataset_usage[item.name] = size_gb
                    total_usage += size_gb
        
        return {
            "total_usage_gb": total_usage,
            "available_gb": self.max_storage_gb - total_usage,
            "datasets": dataset_usage,
            "storage_efficient": total_usage < self.max_storage_gb
        }
    
    def get_download_recommendations(self) -> List[Dict]:
        """Get storage-efficient download recommendations"""
        storage_info = self.check_storage_usage()
        available_gb = storage_info["available_gb"]
        
        recommendations = []
        cumulative_size = 0.0
        
        # Sort by priority
        sorted_datasets = sorted(
            self.recommended_datasets.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for dataset_id, info in sorted_datasets:
            if cumulative_size + info["size_gb"] <= available_gb:
                recommendations.append({
                    "dataset_id": dataset_id,
                    "name": info["name"],
                    "size_gb": info["size_gb"],
                    "priority": info["priority"],
                    "description": info["description"],
                    "fits_in_storage": True
                })
                cumulative_size += info["size_gb"]
            else:
                recommendations.append({
                    "dataset_id": dataset_id,
                    "name": info["name"],
                    "size_gb": info["size_gb"],
                    "priority": info["priority"],
                    "description": info["description"],
                    "fits_in_storage": False
                })
        
        return recommendations
    
    def print_storage_report(self):
        """Print detailed storage report"""
        storage_info = self.check_storage_usage()
        recommendations = self.get_download_recommendations()
        
        print("💾 STORAGE ANALYSIS REPORT")
        print("=" * 50)
        print(f"💿 Total Storage Limit: {self.max_storage_gb:.1f} GB")
        print(f"📊 Current Usage: {storage_info['total_usage_gb']:.1f} GB")
        print(f"🆓 Available Space: {storage_info['available_gb']:.1f} GB")
        print(f"✅ Storage Efficient: {storage_info['storage_efficient']}")
        
        if storage_info["datasets"]:
            print(f"\n📁 Current Datasets:")
            for name, size in storage_info["datasets"].items():
                print(f"  • {name}: {size:.1f} GB")
        
        print(f"\n🎯 RECOMMENDED DOWNLOADS:")
        print("-" * 30)
        
        total_recommended = 0.0
        for rec in recommendations:
            if rec["fits_in_storage"]:
                status = "✅ RECOMMENDED"
                total_recommended += rec["size_gb"]
            else:
                status = "❌ TOO LARGE"
            
            print(f"{status}")
            print(f"  Dataset: {rec['name']}")
            print(f"  Size: {rec['size_gb']:.1f} GB")
            print(f"  Priority: {rec['priority']}")
            print(f"  Description: {rec['description']}")
            print()
        
        print(f"📦 Total Recommended: {total_recommended:.1f} GB")
        print(f"💾 Remaining After Download: {storage_info['available_gb'] - total_recommended:.1f} GB")
    
    def generate_synthetic_dataset(self, 
                                 num_videos: int = 100,
                                 num_audio: int = 200,
                                 target_size_gb: float = 10.0) -> Path:
        """Generate synthetic dataset for development"""
        synthetic_dir = self.data_root / "synthetic_data"
        synthetic_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (synthetic_dir / "real" / "videos").mkdir(parents=True, exist_ok=True)
        (synthetic_dir / "fake" / "videos").mkdir(parents=True, exist_ok=True)
        (synthetic_dir / "real" / "audio").mkdir(parents=True, exist_ok=True)
        (synthetic_dir / "fake" / "audio").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating synthetic dataset (target: {target_size_gb:.1f} GB)")
        
        # Generate synthetic videos (using OpenCV)
        video_files = []
        for i in tqdm(range(num_videos), desc="Generating videos"):
            # Generate fake vs real
            is_fake = i >= num_videos // 2
            category = "fake" if is_fake else "real"
            
            video_path = synthetic_dir / category / "videos" / f"{category}_video_{i:03d}.mp4"
            self._generate_synthetic_video(video_path, is_fake=is_fake)
            video_files.append(str(video_path))
        
        # Generate synthetic audio
        audio_files = []
        for i in tqdm(range(num_audio), desc="Generating audio"):
            is_fake = i >= num_audio // 2
            category = "spoof" if is_fake else "bonafide"
            
            audio_path = synthetic_dir / "fake" if is_fake else synthetic_dir / "real"
            audio_path = audio_path / "audio" / f"{category}_audio_{i:03d}.wav"
            self._generate_synthetic_audio(audio_path, is_fake=is_fake)
            audio_files.append(str(audio_path))
        
        # Create metadata
        metadata = {
            "dataset_type": "synthetic",
            "generation_date": str(Path.cwd()),
            "total_videos": num_videos,
            "total_audio": num_audio,
            "real_videos": num_videos // 2,
            "fake_videos": num_videos // 2,
            "bonafide_audio": num_audio // 2,
            "spoof_audio": num_audio // 2,
            "video_files": video_files,
            "audio_files": audio_files,
            "generation_method": "opencv_procedural"
        }
        
        with open(synthetic_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Check size
        actual_size = sum(f.stat().st_size for f in synthetic_dir.rglob('*') if f.is_file())
        actual_size_gb = actual_size / (1024**3)
        
        logger.info(f"Generated synthetic dataset: {actual_size_gb:.1f} GB at {synthetic_dir}")
        return synthetic_dir
    
    def _generate_synthetic_video(self, output_path: Path, 
                                is_fake: bool = False,
                                duration: float = 3.0,
                                fps: int = 30) -> None:
        """Generate a synthetic video file"""
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_count = int(duration * fps)
        
        # Generate face-like patterns
        if is_fake:
            # "Fake" videos have more artificial patterns
            base_pattern = self._generate_artificial_face_pattern()
        else:
            # "Real" videos have more natural patterns
            base_pattern = self._generate_natural_face_pattern()
        
        with cv2.VideoWriter(str(output_path), fourcc, fps, (224, 224)) as writer:
            for frame_idx in range(frame_count):
                # Add temporal variation
                frame = self._add_temporal_variation(base_pattern, frame_idx, is_fake)
                writer.write(frame)
    
    def _generate_artificial_face_pattern(self) -> np.ndarray:
        """Generate artificial-looking face pattern"""
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Artificial patterns (geometric shapes, high contrast)
        cv2.rectangle(frame, (50, 50), (174, 174), (100, 150, 200), 2)
        cv2.circle(frame, (112, 112), 60, (150, 100, 50), -1)
        cv2.ellipse(frame, (112, 90), (30, 15), 0, 0, 180, (50, 50, 150), -1)  # Eyes region
        cv2.ellipse(frame, (112, 130), (20, 10), 0, 0, 180, (100, 50, 50), -1)  # Mouth region
        
        # Add artificial noise
        noise = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        return frame
    
    def _generate_natural_face_pattern(self) -> np.ndarray:
        """Generate natural-looking face pattern"""
        frame = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
        
        # Natural skin-tone patterns
        cv2.circle(frame, (112, 112), 80, (120, 140, 160), -1)  # Face shape
        cv2.circle(frame, (92, 95), 8, (60, 80, 100), -1)   # Left eye
        cv2.circle(frame, (132, 95), 8, (60, 80, 100), -1)  # Right eye
        cv2.ellipse(frame, (112, 125), (15, 8), 0, 0, 180, (80, 60, 60), -1)  # Mouth
        
        # Smooth with Gaussian blur for natural look
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        return frame
    
    def _add_temporal_variation(self, base_frame: np.ndarray, 
                              frame_idx: int, is_fake: bool) -> np.ndarray:
        """Add temporal variation to frames"""
        frame = base_frame.copy()
        
        # Add movement/variation
        shift_x = int(5 * np.sin(frame_idx * 0.1))
        shift_y = int(3 * np.cos(frame_idx * 0.15))
        
        # Create translation matrix
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(frame, M, (224, 224))
        
        if is_fake:
            # Add artificial artifacts for "fake" videos
            if frame_idx % 10 == 0:  # Every 10th frame
                # Add compression artifacts
                frame = cv2.resize(frame, (56, 56))
                frame = cv2.resize(frame, (224, 224))
        
        return frame
    
    def _generate_synthetic_audio(self, output_path: Path,
                                is_fake: bool = False,
                                duration: float = 3.0,
                                sample_rate: int = 16000) -> None:
        """Generate synthetic audio file"""
        # Generate synthetic audio data
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        if is_fake:
            # "Fake" audio with artificial patterns
            audio = self._generate_artificial_speech(t)
        else:
            # "Real" audio with natural speech patterns
            audio = self._generate_natural_speech(t)
        
        # Save as WAV file
        try:
            import soundfile as sf
            sf.write(str(output_path), audio, sample_rate)
        except ImportError:
            # Fallback to simple wave writing
            self._write_wav_simple(output_path, audio, sample_rate)
    
    def _generate_artificial_speech(self, t: np.ndarray) -> np.ndarray:
        """Generate artificial-sounding speech"""
        # Synthetic speech with robotic characteristics
        fundamental = 150  # Hz
        audio = 0.3 * np.sin(2 * np.pi * fundamental * t)
        audio += 0.2 * np.sin(2 * np.pi * fundamental * 2 * t)  # Harmonic
        audio += 0.1 * np.sin(2 * np.pi * fundamental * 3 * t)  # Harmonic
        
        # Add robotic modulation
        modulation = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
        audio = audio * modulation
        
        # Add digital artifacts
        audio = np.clip(audio, -0.8, 0.8)  # Hard clipping
        
        return audio.astype(np.float32)
    
    def _generate_natural_speech(self, t: np.ndarray) -> np.ndarray:
        """Generate natural-sounding speech"""
        # More natural speech patterns
        audio = np.zeros_like(t)
        
        # Multiple formants (speech resonances)
        formants = [400, 800, 1200, 1600]  # Typical speech formants
        for i, formant in enumerate(formants):
            amplitude = 0.2 / (i + 1)  # Decreasing amplitude
            audio += amplitude * np.sin(2 * np.pi * formant * t)
        
        # Natural amplitude modulation (speech envelope)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t)) * np.exp(-t/2)
        audio = audio * envelope
        
        # Add natural noise
        audio += 0.02 * np.random.normal(0, 1, len(t))
        
        return audio.astype(np.float32)
    
    def _write_wav_simple(self, filepath: Path, audio: np.ndarray, sample_rate: int):
        """Simple WAV file writer (fallback)"""
        import struct
        
        # Normalize audio to 16-bit range
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with open(filepath, 'wb') as f:
            # WAV header
            f.write(b'RIFF')
            f.write(struct.pack('<L', len(audio_int16) * 2 + 36))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write(struct.pack('<L', 16))
            f.write(struct.pack('<H', 1))  # PCM
            f.write(struct.pack('<H', 1))  # Mono
            f.write(struct.pack('<L', sample_rate))
            f.write(struct.pack('<L', sample_rate * 2))
            f.write(struct.pack('<H', 2))
            f.write(struct.pack('<H', 16))
            f.write(b'data')
            f.write(struct.pack('<L', len(audio_int16) * 2))
            
            # Audio data
            for sample in audio_int16:
                f.write(struct.pack('<h', sample))


def main():
    """Demo storage-efficient dataset management"""
    print("💾 STORAGE-EFFICIENT DATASET MANAGER")
    print("=" * 60)
    
    # Initialize with 50GB limit (adjust based on your needs)
    manager = StorageEfficientDatasetManager(max_storage_gb=50.0)
    
    # Print storage report
    manager.print_storage_report()
    
    # Generate synthetic dataset
    print(f"\n🤖 GENERATING SYNTHETIC DATASET")
    print("-" * 40)
    synthetic_dir = manager.generate_synthetic_dataset(
        num_videos=20,  # Small for demo
        num_audio=40,   # Small for demo
        target_size_gb=1.0  # 1GB for demo
    )
    
    # Updated storage report
    print(f"\n💾 UPDATED STORAGE REPORT")
    print("-" * 40)
    manager.print_storage_report()
    
    print(f"\n✅ STORAGE-EFFICIENT SETUP COMPLETE!")
    print(f"📁 Synthetic dataset created at: {synthetic_dir}")
    print(f"🎯 Ready for development with minimal storage usage")


if __name__ == "__main__":
    main()
