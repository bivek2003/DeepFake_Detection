"""
Dataset Management for Deepfake Detection

Handles dataset information, downloads, and organization for multiple
deepfake detection datasets including FaceForensics++, DFDC, Celeb-DF, etc.

Author: Bivek Sharma Panthi
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a deepfake dataset"""
    name: str
    type: str  # 'video', 'audio', 'multimodal'
    url: str
    description: str
    requires_form: bool = False
    file_count: Optional[int] = None
    size_gb: Optional[float] = None


class DatasetRegistry:
    """Registry of available deepfake detection datasets"""
    
    def __init__(self):
        self.datasets = {
            "faceforensics": DatasetInfo(
                name="FaceForensics++",
                type="video",
                url="https://github.com/ondyari/FaceForensics",
                description="1,000 videos, 1.8M manipulated images",
                requires_form=True,
                file_count=1000,
                size_gb=38.5
            ),
            "dfdc": DatasetInfo(
                name="DFDC (Deepfake Detection Challenge)",
                type="video", 
                url="https://ai.facebook.com/datasets/dfdc/",
                description="100K+ clips, 3,426 actors",
                requires_form=True,
                file_count=100000,
                size_gb=470.0
            ),
            "celebdf": DatasetInfo(
                name="Celeb-DF",
                type="video",
                url="https://github.com/yuezunli/celeb-deepfakeforensics",
                description="590 originals + 5,639 deepfakes",
                requires_form=True,
                file_count=6229,
                size_gb=15.8
            ),
            "wilddeepfake": DatasetInfo(
                name="WildDeepfake", 
                type="video",
                url="https://huggingface.co/datasets/wilddeepfake",
                description="707 real 'in-the-wild' deepfake videos",
                requires_form=False,
                file_count=707,
                size_gb=4.2
            ),
            "deeperforensics": DatasetInfo(
                name="DeeperForensics-1.0",
                type="video",
                url="https://github.com/EndlessSora/DeeperForensics-1.0",
                description="60K videos (17.6M frames), rich perturbations",
                requires_form=True,
                file_count=60000,
                size_gb=2000.0
            ),
            "asvspoof2021": DatasetInfo(
                name="ASVspoof 2021",
                type="audio",
                url="https://zenodo.org/record/4837263",
                description="TTS/VC speech deepfake dataset",
                requires_form=False,
                file_count=611829,
                size_gb=23.1
            ),
            "fakeavceleb": DatasetInfo(
                name="FakeAVCeleb",
                type="multimodal",
                url="https://openreview.net/forum?id=TAXFsg6ZaOl",
                description="Synchronized fake video + lip-synced fake audio",
                requires_form=True,
                file_count=20000,
                size_gb=87.5
            )
        }
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        """Get dataset information by ID"""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self, dataset_type: Optional[str] = None) -> List[DatasetInfo]:
        """List all datasets, optionally filtered by type"""
        datasets = list(self.datasets.values())
        if dataset_type:
            datasets = [d for d in datasets if d.type == dataset_type]
        return datasets
    
    def get_total_size(self, dataset_ids: List[str]) -> float:
        """Calculate total size in GB for selected datasets"""
        total = 0.0
        for dataset_id in dataset_ids:
            dataset = self.get_dataset(dataset_id)
            if dataset and dataset.size_gb:
                total += dataset.size_gb
        return total


class DatasetManager:
    """Manages dataset downloads, verification, and organization"""
    
    def __init__(self, data_root: str = "./datasets"):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.registry = DatasetRegistry()
        
        # Create metadata directory
        self.metadata_dir = self.data_root / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized DatasetManager with root: {self.data_root}")
    
    def print_dataset_info(self, detailed: bool = False):
        """Print information about available datasets"""
        print("🎬 Available Deepfake Detection Datasets")
        print("=" * 60)
        
        # Group by type
        by_type = {}
        for dataset in self.registry.list_datasets():
            if dataset.type not in by_type:
                by_type[dataset.type] = []
            by_type[dataset.type].append(dataset)
        
        for dataset_type, datasets in by_type.items():
            print(f"\n📹 {dataset_type.upper()} DATASETS:")
            print("-" * 40)
            
            for dataset in datasets:
                print(f"• {dataset.name}")
                if detailed:
                    print(f"  Description: {dataset.description}")
                    print(f"  URL: {dataset.url}")
                    if dataset.size_gb:
                        print(f"  Size: {dataset.size_gb:.1f} GB")
                    if dataset.file_count:
                        print(f"  Files: {dataset.file_count:,}")
                    if dataset.requires_form:
                        print("  ⚠️  Requires form submission for access")
                print()
        
        total_size = sum(d.size_gb for d in self.registry.list_datasets() if d.size_gb)
        print(f"📊 Total dataset size: {total_size:.1f} GB")
    
    def create_directory_structure(self, dataset_id: str) -> Path:
        """Create directory structure for a dataset"""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        
        dataset_dir = self.data_root / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        
        # Create standard subdirectories based on dataset type
        if dataset.type == "video":
            (dataset_dir / "real").mkdir(exist_ok=True)
            (dataset_dir / "fake").mkdir(exist_ok=True)
            (dataset_dir / "metadata").mkdir(exist_ok=True)
        elif dataset.type == "audio":
            (dataset_dir / "bonafide").mkdir(exist_ok=True)  # Real audio
            (dataset_dir / "spoof").mkdir(exist_ok=True)     # Fake audio  
            (dataset_dir / "metadata").mkdir(exist_ok=True)
        elif dataset.type == "multimodal":
            (dataset_dir / "real_video_real_audio").mkdir(exist_ok=True)
            (dataset_dir / "fake_video_real_audio").mkdir(exist_ok=True)
            (dataset_dir / "real_video_fake_audio").mkdir(exist_ok=True)
            (dataset_dir / "fake_video_fake_audio").mkdir(exist_ok=True)
            (dataset_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"Created directory structure for {dataset.name}")
        return dataset_dir
    
    def save_dataset_metadata(self, dataset_id: str, metadata: Dict):
        """Save metadata for a dataset"""
        metadata_file = self.metadata_dir / f"{dataset_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata for {dataset_id}")
    
    def load_dataset_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Load metadata for a dataset"""
        metadata_file = self.metadata_dir / f"{dataset_id}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def verify_dataset(self, dataset_id: str) -> Dict[str, any]:
        """Verify dataset integrity and completeness"""
        dataset_dir = self.data_root / dataset_id
        if not dataset_dir.exists():
            return {"exists": False, "error": "Dataset directory not found"}
        
        verification = {
            "exists": True,
            "dataset_id": dataset_id,
            "path": str(dataset_dir),
            "subdirectories": [],
            "file_count": 0,
            "total_size_mb": 0.0
        }
        
        # Count files and calculate size
        for item in dataset_dir.rglob("*"):
            if item.is_file():
                verification["file_count"] += 1
                verification["total_size_mb"] += item.stat().st_size / (1024 * 1024)
            elif item.is_dir():
                rel_path = item.relative_to(dataset_dir)
                verification["subdirectories"].append(str(rel_path))
        
        return verification
    
    def create_sample_dataset(self, name: str = "sample_data", 
                            video_count: int = 10, audio_count: int = 10) -> Path:
        """Create a sample dataset structure for testing"""
        sample_dir = self.data_root / name
        sample_dir.mkdir(exist_ok=True)
        
        # Video structure
        video_dir = sample_dir / "video"
        (video_dir / "real").mkdir(parents=True, exist_ok=True)
        (video_dir / "fake").mkdir(parents=True, exist_ok=True)
        
        # Audio structure  
        audio_dir = sample_dir / "audio"
        (audio_dir / "bonafide").mkdir(parents=True, exist_ok=True)
        (audio_dir / "spoof").mkdir(parents=True, exist_ok=True)
        
        # Create placeholder files for demonstration
        file_list = []
        
        # Video files
        for i in range(video_count):
            real_file = video_dir / "real" / f"real_video_{i:03d}.mp4"
            fake_file = video_dir / "fake" / f"fake_video_{i:03d}.mp4"
            file_list.extend([str(real_file), str(fake_file)])
        
        # Audio files
        for i in range(audio_count):
            real_file = audio_dir / "bonafide" / f"real_audio_{i:03d}.wav"  
            fake_file = audio_dir / "spoof" / f"fake_audio_{i:03d}.wav"
            file_list.extend([str(real_file), str(fake_file)])
        
        # Save file list metadata
        metadata = {
            "name": name,
            "created_at": str(Path.cwd()),
            "video_count": video_count * 2,  # real + fake
            "audio_count": audio_count * 2,  # real + fake
            "file_list": file_list,
            "structure": "sample dataset for development/testing"
        }
        
        metadata_file = sample_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created sample dataset at: {sample_dir}")
        return sample_dir
    
    def list_local_datasets(self) -> List[Dict[str, any]]:
        """List all datasets present locally"""
        local_datasets = []
        
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name != "metadata":
                verification = self.verify_dataset(item.name)
                dataset_info = self.registry.get_dataset(item.name)
                
                local_dataset = {
                    "id": item.name,
                    "path": str(item),
                    "verification": verification,
                    "registry_info": dataset_info
                }
                local_datasets.append(local_dataset)
        
        return local_datasets
    
    def get_download_instructions(self, dataset_id: str) -> str:
        """Get download instructions for a dataset"""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            return f"Unknown dataset: {dataset_id}"
        
        instructions = f"""
📋 Download Instructions for {dataset.name}
{'=' * 50}

Dataset: {dataset.name}
Type: {dataset.type}
Description: {dataset.description}
Size: {dataset.size_gb:.1f} GB ({dataset.file_count:,} files)

URL: {dataset.url}

"""
        
        if dataset.requires_form:
            instructions += """
⚠️  FORM SUBMISSION REQUIRED
This dataset requires filling out a request form before access.
Please visit the URL above and follow their access procedure.

"""
        
        instructions += f"""
📁 Local Directory Structure:
The dataset should be placed in: {self.data_root / dataset_id}

After download, run:
python -c "from deepfake_detector.data import DatasetManager; dm = DatasetManager(); dm.verify_dataset('{dataset_id}')"
"""
        
        return instructions


def main():
    """Demonstration of dataset management functionality"""
    # Initialize dataset manager
    dm = DatasetManager()
    
    # Print available datasets
    dm.print_dataset_info(detailed=True)
    
    # Create sample dataset
    sample_dir = dm.create_sample_dataset()
    
    # List local datasets
    print("\n📂 Local Datasets:")
    local_datasets = dm.list_local_datasets()
    for dataset in local_datasets:
        print(f"• {dataset['id']}: {dataset['verification']['file_count']} files")
    
    # Show download instructions
    print(dm.get_download_instructions("faceforensics"))


if __name__ == "__main__":
    main()
