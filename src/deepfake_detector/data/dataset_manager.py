"""
Dataset Management for Deepfake Detection

Handles dataset information, downloads, and organization for multiple
deepfake detection datasets including FaceForensics++, DFDC, Celeb-DF, etc.
Optimized for storage-efficient development with 260GB constraints.

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
    storage_priority: int = 3  # 1=high priority, 5=low priority
    compressed_available: bool = False
    compressed_size_gb: Optional[float] = None
    eval_only_size_gb: Optional[float] = None


class DatasetRegistry:
    """Registry of available deepfake detection datasets with storage optimization"""
    
    def __init__(self):
        self.datasets = {
            "faceforensics": DatasetInfo(
                name="FaceForensics++",
                type="video",
                url="https://github.com/ondyari/FaceForensics",
                description="1,000 videos, 1.8M manipulated images",
                requires_form=True,
                file_count=1000,
                size_gb=38.5,
                storage_priority=3,  # Medium-low priority due to size
                compressed_available=True,
                compressed_size_gb=15.0
            ),
            "dfdc": DatasetInfo(
                name="DFDC (Deepfake Detection Challenge)",
                type="video", 
                url="https://ai.facebook.com/datasets/dfdc/",
                description="100K+ clips, 3,426 actors",
                requires_form=True,
                file_count=100000,
                size_gb=470.0,
                storage_priority=5,  # Lowest priority (too large for 260GB)
                compressed_available=False
            ),
            "celebdf": DatasetInfo(
                name="Celeb-DF",
                type="video",
                url="https://github.com/yuezunli/celeb-deepfakeforensics",
                description="590 originals + 5,639 deepfakes",
                requires_form=True,
                file_count=6229,
                size_gb=15.8,
                storage_priority=2,  # Medium priority
                compressed_available=True,
                compressed_size_gb=8.0
            ),
            "wilddeepfake": DatasetInfo(
                name="WildDeepfake", 
                type="video",
                url="https://huggingface.co/datasets/wilddeepfake",
                description="707 real 'in-the-wild' deepfake videos",
                requires_form=False,
                file_count=707,
                size_gb=4.2,
                storage_priority=1,  # Highest priority (small, accessible)
                compressed_available=False
            ),
            "deeperforensics": DatasetInfo(
                name="DeeperForensics-1.0",
                type="video",
                url="https://github.com/EndlessSora/DeeperForensics-1.0",
                description="60K videos (17.6M frames), rich perturbations",
                requires_form=True,
                file_count=60000,
                size_gb=2000.0,
                storage_priority=5,  # Lowest priority (massive dataset)
                compressed_available=False
            ),
            "asvspoof2021": DatasetInfo(
                name="ASVspoof 2021",
                type="audio",
                url="https://zenodo.org/record/4837263",
                description="TTS/VC speech deepfake dataset",
                requires_form=False,
                file_count=611829,
                size_gb=23.1,
                storage_priority=2,  # Medium priority
                compressed_available=True,
                compressed_size_gb=8.0,
                eval_only_size_gb=3.5  # Just evaluation set
            ),
            "fakeavceleb": DatasetInfo(
                name="FakeAVCeleb",
                type="multimodal",
                url="https://openreview.net/forum?id=TAXFsg6ZaOl",
                description="Synchronized fake video + lip-synced fake audio",
                requires_form=True,
                file_count=20000,
                size_gb=87.5,
                storage_priority=4,  # Low priority due to size
                compressed_available=False
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
    
    def get_storage_recommendations(self, storage_limit_gb: float = 50.0) -> List[DatasetInfo]:
        """Get storage-optimized dataset recommendations"""
        # Sort by storage priority (1 = highest priority)
        sorted_datasets = sorted(self.datasets.values(), key=lambda x: x.storage_priority)
        
        recommendations = []
        cumulative_size = 0.0
        
        for dataset in sorted_datasets:
            # Check if compressed version fits better
            if dataset.compressed_available and dataset.compressed_size_gb:
                size_to_check = dataset.compressed_size_gb
            elif dataset.eval_only_size_gb:
                size_to_check = dataset.eval_only_size_gb
            else:
                size_to_check = dataset.size_gb or 0
            
            if cumulative_size + size_to_check <= storage_limit_gb:
                recommendations.append(dataset)
                cumulative_size += size_to_check
            
            # Stop when we reach reasonable coverage
            if len(recommendations) >= 4 and cumulative_size > storage_limit_gb * 0.8:
                break
        
        return recommendations


class DatasetManager:
    """Manages dataset downloads, verification, and organization with storage optimization"""
    
    def __init__(self, data_root: str = "./datasets", storage_limit_gb: float = 50.0):
        self.data_root = Path(data_root)
        self.data_root.mkdir(exist_ok=True)
        self.registry = DatasetRegistry()
        self.storage_limit_gb = storage_limit_gb
        
        # Create metadata directory
        self.metadata_dir = self.data_root / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized DatasetManager with root: {self.data_root}")
        logger.info(f"Storage limit: {storage_limit_gb:.1f} GB")
    
    def check_storage_usage(self) -> Dict[str, float]:
        """Check current storage usage"""
        total_usage = 0.0
        dataset_usage = {}
        
        if self.data_root.exists():
            for item in self.data_root.iterdir():
                if item.is_dir() and item.name != "metadata":
                    size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    size_gb = size_bytes / (1024**3)
                    dataset_usage[item.name] = size_gb
                    total_usage += size_gb
        
        return {
            "total_usage_gb": total_usage,
            "available_gb": self.storage_limit_gb - total_usage,
            "datasets": dataset_usage,
            "within_limit": total_usage <= self.storage_limit_gb
        }
    
    def print_dataset_info(self, detailed: bool = False):
        """Print information about available datasets with storage considerations"""
        storage_info = self.check_storage_usage()
        
        print("🎬 Available Deepfake Detection Datasets")
        print("=" * 60)
        print(f"💾 Storage Limit: {self.storage_limit_gb:.1f} GB")
        print(f"📊 Current Usage: {storage_info['total_usage_gb']:.1f} GB")
        print(f"🆓 Available Space: {storage_info['available_gb']:.1f} GB")
        
        # Group by type and sort by storage priority
        by_type = {}
        for dataset in self.registry.list_datasets():
            if dataset.type not in by_type:
                by_type[dataset.type] = []
            by_type[dataset.type].append(dataset)
        
        # Sort each type by storage priority
        for dataset_type in by_type:
            by_type[dataset_type].sort(key=lambda x: x.storage_priority)
        
        total_recommended = 0
        for dataset_type, datasets in by_type.items():
            print(f"\n📹 {dataset_type.upper()} DATASETS:")
            print("-" * 40)
            
            for dataset in datasets:
                size_gb = dataset.size_gb or 0
                priority = dataset.storage_priority
                
                # Determine recommendation based on size and priority
                if size_gb <= 5 and priority <= 2:
                    recommendation = "✅ RECOMMENDED"
                    total_recommended += size_gb
                elif size_gb <= 15 and priority <= 3:
                    recommendation = "⚠️  CONSIDER"
                elif size_gb > 100:
                    recommendation = "❌ TOO LARGE"
                else:
                    recommendation = "🔍 EVALUATE"
                
                print(f"{recommendation} {dataset.name}")
                
                if detailed:
                    print(f"  Description: {dataset.description}")
                    print(f"  URL: {dataset.url}")
                    print(f"  Size: {size_gb:.1f} GB (Priority {priority}/5)")
                    
                    if dataset.compressed_available and dataset.compressed_size_gb:
                        print(f"  💾 Compressed: {dataset.compressed_size_gb:.1f} GB available")
                    
                    if dataset.eval_only_size_gb:
                        print(f"  📊 Eval only: {dataset.eval_only_size_gb:.1f} GB")
                    
                    if dataset.requires_form:
                        print("  ⚠️  Requires form submission for access")
                    
                    if not dataset.requires_form:
                        print("  ✅ Direct download available")
                else:
                    print(f"  Size: {size_gb:.1f} GB | Priority: {priority}/5")
                
                print()
        
        # Storage recommendations
        print(f"💡 STORAGE-EFFICIENT STRATEGY:")
        recommendations = self.registry.get_storage_recommendations(self.storage_limit_gb)
        
        print(f"\n🎯 TOP RECOMMENDATIONS (fits in {self.storage_limit_gb:.1f} GB):")
        cumulative = 0
        for i, rec in enumerate(recommendations, 1):
            size = rec.compressed_size_gb if rec.compressed_available and rec.compressed_size_gb else rec.size_gb or 0
            cumulative += size
            version = " (compressed)" if rec.compressed_available and rec.compressed_size_gb else ""
            form_req = " - Requires form" if rec.requires_form else " - Direct download"
            
            print(f"  {i}. {rec.name}: {size:.1f} GB{version}{form_req}")
        
        print(f"\n📦 Total recommended: {cumulative:.1f} GB")
        print(f"💾 Remaining space: {self.storage_limit_gb - cumulative:.1f} GB")
        
        print(f"\n🚀 DEVELOPMENT STRATEGY:")
        print("  • Start with WildDeepfake (4.2 GB, no forms)")
        print("  • Add ASVspoof eval set (3.5 GB)")
        print("  • Generate synthetic data (10-15 GB)")
        print("  • Consider Celeb-DF subset if space allows")
        print("  • Total development dataset: ~30-40 GB")
    
    def get_download_instructions(self, dataset_id: str, storage_optimized: bool = True) -> str:
        """Get download instructions for a dataset with storage considerations"""
        dataset = self.registry.get_dataset(dataset_id)
        if not dataset:
            return f"Unknown dataset: {dataset_id}"
        
        storage_info = self.check_storage_usage()
        
        instructions = f"""
📋 Download Instructions for {dataset.name}
{'=' * 60}

Dataset: {dataset.name}
Type: {dataset.type}
Description: {dataset.description}
Full Size: {dataset.size_gb:.1f} GB
Priority: {dataset.storage_priority}/5 (1=highest)

URL: {dataset.url}

💾 STORAGE CONSIDERATIONS:
Available Space: {storage_info['available_gb']:.1f} GB
"""
        
        # Storage-optimized recommendations
        if storage_optimized:
            instructions += f"\n🎯 STORAGE-OPTIMIZED OPTIONS:\n"
            
            if dataset.compressed_available and dataset.compressed_size_gb:
                fits = dataset.compressed_size_gb <= storage_info['available_gb']
                status = "✅ FITS" if fits else "❌ TOO LARGE"
                instructions += f"• Compressed version: {dataset.compressed_size_gb:.1f} GB [{status}]\n"
            
            if dataset.eval_only_size_gb:
                fits = dataset.eval_only_size_gb <= storage_info['available_gb']
                status = "✅ FITS" if fits else "❌ TOO LARGE"
                instructions += f"• Evaluation set only: {dataset.eval_only_size_gb:.1f} GB [{status}]\n"
            
            full_fits = (dataset.size_gb or 0) <= storage_info['available_gb']
            status = "✅ FITS" if full_fits else "❌ TOO LARGE"
            instructions += f"• Full dataset: {dataset.size_gb:.1f} GB [{status}]\n"
        
        if dataset.requires_form:
            instructions += f"""
⚠️  FORM SUBMISSION REQUIRED
This dataset requires filling out a request form before access.
Please visit the URL above and follow their access procedure.
"""
        else:
            instructions += f"""
✅ DIRECT DOWNLOAD AVAILABLE
No form submission required for this dataset.
"""
        
        instructions += f"""
📁 Local Directory Structure:
The dataset should be placed in: {self.data_root / dataset_id}

After download, run:
python -c "from deepfake_detector.data import DatasetManager; dm = DatasetManager(); print(dm.verify_dataset('{dataset_id}'))"

🎯 RECOMMENDATION: 
Priority {dataset.storage_priority}/5 for {self.storage_limit_gb:.1f}GB storage limit
"""
        
        return instructions
    
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
        
        # Add storage information
        storage_info = self.check_storage_usage()
        metadata.update({
            "storage_info": storage_info,
            "storage_limit_gb": self.storage_limit_gb,
            "dataset_registry_info": self.registry.get_dataset(dataset_id).__dict__ if self.registry.get_dataset(dataset_id) else None
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
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
            "total_size_mb": 0.0,
            "total_size_gb": 0.0
        }
        
        # Count files and calculate size
        for item in dataset_dir.rglob("*"):
            if item.is_file():
                verification["file_count"] += 1
                size_bytes = item.stat().st_size
                verification["total_size_mb"] += size_bytes / (1024 * 1024)
            elif item.is_dir():
                rel_path = item.relative_to(dataset_dir)
                verification["subdirectories"].append(str(rel_path))
        
        verification["total_size_gb"] = verification["total_size_mb"] / 1024
        
        # Add registry information
        dataset_info = self.registry.get_dataset(dataset_id)
        if dataset_info:
            verification["registry_info"] = {
                "expected_size_gb": dataset_info.size_gb,
                "storage_priority": dataset_info.storage_priority,
                "compressed_available": dataset_info.compressed_available
            }
            
            # Check if size matches expectations
            if dataset_info.size_gb:
                size_ratio = verification["total_size_gb"] / dataset_info.size_gb
                verification["size_check"] = {
                    "expected_gb": dataset_info.size_gb,
                    "actual_gb": verification["total_size_gb"],
                    "ratio": size_ratio,
                    "reasonable": 0.8 <= size_ratio <= 1.2  # Within 20% is reasonable
                }
        
        return verification
    
    def create_sample_dataset(self, name: str = "sample_data", 
                            video_count: int = 10, audio_count: int = 10) -> Path:
        """Create a sample dataset structure for testing (storage-efficient)"""
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
        
        # Save file list metadata with storage info
        storage_info = self.check_storage_usage()
        metadata = {
            "name": name,
            "created_at": str(Path.cwd()),
            "video_count": video_count * 2,  # real + fake
            "audio_count": audio_count * 2,  # real + fake
            "file_list": file_list,
            "structure": "sample dataset for development/testing",
            "storage_efficient": True,
            "storage_info": storage_info,
            "storage_limit_gb": self.storage_limit_gb
        }
        
        metadata_file = sample_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Created sample dataset at: {sample_dir}")
        return sample_dir
    
    def list_local_datasets(self) -> List[Dict[str, any]]:
        """List all datasets present locally with storage information"""
        local_datasets = []
        
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name != "metadata":
                verification = self.verify_dataset(item.name)
                dataset_info = self.registry.get_dataset(item.name)
                
                local_dataset = {
                    "id": item.name,
                    "path": str(item),
                    "verification": verification,
                    "registry_info": dataset_info,
                    "size_gb": verification.get("total_size_gb", 0),
                    "storage_efficient": verification.get("total_size_gb", 0) <= 20  # Mark as efficient if ≤20GB
                }
                local_datasets.append(local_dataset)
        
        # Sort by size (smallest first)
        local_datasets.sort(key=lambda x: x["size_gb"])
        
        return local_datasets
    
    def print_storage_summary(self):
        """Print comprehensive storage summary"""
        storage_info = self.check_storage_usage()
        local_datasets = self.list_local_datasets()
        
        print("💾 STORAGE SUMMARY REPORT")
        print("=" * 50)
        print(f"📊 Storage Limit: {self.storage_limit_gb:.1f} GB")
        print(f"📈 Current Usage: {storage_info['total_usage_gb']:.1f} GB ({storage_info['total_usage_gb']/self.storage_limit_gb*100:.1f}%)")
        print(f"🆓 Available Space: {storage_info['available_gb']:.1f} GB")
        print(f"✅ Within Limit: {'Yes' if storage_info['within_limit'] else 'No'}")
        
        if local_datasets:
            print(f"\n📁 Local Datasets ({len(local_datasets)}):")
            for dataset in local_datasets:
                efficient_icon = "💾" if dataset["storage_efficient"] else "⚠️"
                print(f"  {efficient_icon} {dataset['id']}: {dataset['size_gb']:.1f} GB")
        
        recommendations = self.registry.get_storage_recommendations(storage_info['available_gb'])
        if recommendations:
            print(f"\n🎯 Can Still Download:")
            for rec in recommendations[:3]:  # Top 3
                size = rec.compressed_size_gb if rec.compressed_available and rec.compressed_size_gb else rec.size_gb or 0
                print(f"  • {rec.name}: {size:.1f} GB")


def main():
    """Demonstration of storage-efficient dataset management functionality"""
    # Initialize dataset manager with 50GB limit (adjust as needed)
    dm = DatasetManager(storage_limit_gb=50.0)
    
    print("🎬 STORAGE-EFFICIENT DATASET MANAGEMENT DEMO")
    print("=" * 60)
    
    # Print available datasets with storage considerations
    dm.print_dataset_info(detailed=True)
    
    # Create sample dataset
    sample_dir = dm.create_sample_dataset("demo_data", video_count=5, audio_count=5)
    
    # Print storage summary
    print("\n" + "="*60)
    dm.print_storage_summary()
    
    # Show download instructions for recommended dataset
    print("\n" + "="*60)
    print(dm.get_download_instructions("wilddeepfake"))
    
    print(f"\n✅ STORAGE-EFFICIENT SETUP COMPLETE!")
    print(f"🎯 Perfect for 260GB storage constraint")
    print(f"🚀 Ready for professional ML development")


if __name__ == "__main__":
    main()
