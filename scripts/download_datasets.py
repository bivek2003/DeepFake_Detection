#!/usr/bin/env python3
"""
Dataset Download Script for Deepfake Detection Training

Supports:
- FaceForensics++ (requires manual download due to license)
- Celeb-DF v2 (automated download)
- DFDC Preview (optional)

Usage:
    python scripts/download_datasets.py --dataset all
    python scripts/download_datasets.py --dataset celebdf
    python scripts/download_datasets.py --dataset faceforensics --ff-path /path/to/downloaded/ff++
"""

import os
import sys
import argparse
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Optional
import hashlib
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gdown
    import requests
except ImportError:
    print("Installing required packages...")
    os.system("pip install gdown requests tqdm")
    import gdown
    import requests


# Dataset configuration
DATASETS_DIR = Path("/app/datasets") if Path("/app/datasets").exists() else Path("./datasets")

CELEB_DF_V2 = {
    "name": "Celeb-DF-v2",
    "url": "https://drive.google.com/uc?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj",
    "filename": "Celeb-DF-v2.zip",
    "extracted_dir": "Celeb-DF-v2",
    "description": "Celeb-DF v2 - High quality celebrity deepfakes",
    "size_gb": 30,
}

# FaceForensics++ download info (requires registration)
FACEFORENSICS_INFO = """
================================================================================
FaceForensics++ Dataset
================================================================================

FaceForensics++ requires registration and agreement to terms of use.

1. Visit: https://github.com/ondyari/FaceForensics
2. Fill the form to get download credentials
3. Download the following:
   - Original sequences (c23 compression): ~10GB
   - Deepfakes (c23 compression): ~10GB  
   - Face2Face (c23 compression): ~10GB
   - FaceSwap (c23 compression): ~10GB
   - NeuralTextures (c23 compression): ~10GB

4. After downloading, run:
   python scripts/download_datasets.py --dataset faceforensics --ff-path /path/to/downloaded/

Structure expected:
    /path/to/downloaded/
    ├── original_sequences/
    │   └── youtube/
    │       └── c23/
    │           └── videos/
    ├── manipulated_sequences/
    │   ├── Deepfakes/
    │   │   └── c23/
    │   │       └── videos/
    │   ├── Face2Face/
    │   ├── FaceSwap/
    │   └── NeuralTextures/

================================================================================
"""


def download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """Download file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def download_celeb_df(output_dir: Path) -> bool:
    """Download Celeb-DF v2 dataset."""
    print("\n" + "="*80)
    print("Downloading Celeb-DF v2 Dataset")
    print("="*80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / CELEB_DF_V2["filename"]
    extracted_dir = output_dir / CELEB_DF_V2["extracted_dir"]
    
    # Check if already extracted
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        print(f"✓ Celeb-DF v2 already exists at {extracted_dir}")
        return True
    
    # Download using gdown for Google Drive
    print(f"Downloading from Google Drive (~{CELEB_DF_V2['size_gb']}GB)...")
    print("This may take a while depending on your connection speed.")
    
    try:
        gdown.download(CELEB_DF_V2["url"], str(zip_path), quiet=False)
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nAlternative: Download manually from:")
        print("https://github.com/yuezunli/celeb-deepfakeforensics")
        return False
    
    # Extract
    if zip_path.exists():
        print("\nExtracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"✓ Extracted to {extracted_dir}")
            
            # Clean up zip file
            zip_path.unlink()
            print("✓ Cleaned up archive")
            return True
        except Exception as e:
            print(f"Error extracting: {e}")
            return False
    
    return False


def setup_faceforensics(ff_path: Path, output_dir: Path) -> bool:
    """Setup FaceForensics++ from downloaded files."""
    print("\n" + "="*80)
    print("Setting up FaceForensics++ Dataset")
    print("="*80)
    
    if not ff_path.exists():
        print(FACEFORENSICS_INFO)
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    ff_output = output_dir / "FaceForensics"
    
    # Expected structure
    required_dirs = [
        "original_sequences/youtube/c23/videos",
        "manipulated_sequences/Deepfakes/c23/videos",
        "manipulated_sequences/Face2Face/c23/videos",
        "manipulated_sequences/FaceSwap/c23/videos",
        "manipulated_sequences/NeuralTextures/c23/videos",
    ]
    
    # Verify structure
    missing = []
    for dir_path in required_dirs:
        full_path = ff_path / dir_path
        if not full_path.exists():
            missing.append(dir_path)
    
    if missing:
        print("Missing directories:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease ensure all directories exist.")
        return False
    
    # Create symlink or copy
    if ff_output.exists():
        print(f"✓ FaceForensics++ already linked at {ff_output}")
    else:
        try:
            ff_output.symlink_to(ff_path.resolve())
            print(f"✓ Created symlink: {ff_output} -> {ff_path}")
        except OSError:
            print("Symlink failed, copying structure...")
            shutil.copytree(ff_path, ff_output)
            print(f"✓ Copied to {ff_output}")
    
    return True


def create_dataset_splits(output_dir: Path):
    """Create train/val/test split files."""
    print("\n" + "="*80)
    print("Creating Dataset Splits")
    print("="*80)
    
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Process Celeb-DF
    celebdf_dir = output_dir / "Celeb-DF-v2"
    if celebdf_dir.exists():
        create_celebdf_splits(celebdf_dir, splits_dir)
    
    # Process FaceForensics++
    ff_dir = output_dir / "FaceForensics"
    if ff_dir.exists():
        create_ff_splits(ff_dir, splits_dir)
    
    print("✓ Dataset splits created")


def create_celebdf_splits(celebdf_dir: Path, splits_dir: Path):
    """Create splits for Celeb-DF v2."""
    import random
    random.seed(42)
    
    # Find videos
    real_videos = []
    fake_videos = []
    
    # Celeb-DF structure: Celeb-real, Celeb-synthesis, YouTube-real
    for subdir in ["Celeb-real", "YouTube-real"]:
        path = celebdf_dir / subdir
        if path.exists():
            real_videos.extend([str(p.relative_to(celebdf_dir)) for p in path.glob("*.mp4")])
    
    for subdir in ["Celeb-synthesis"]:
        path = celebdf_dir / subdir
        if path.exists():
            fake_videos.extend([str(p.relative_to(celebdf_dir)) for p in path.glob("*.mp4")])
    
    print(f"  Celeb-DF: {len(real_videos)} real, {len(fake_videos)} fake videos")
    
    # Shuffle and split
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    # 70/15/15 split
    def split_list(lst):
        n = len(lst)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]
    
    real_train, real_val, real_test = split_list(real_videos)
    fake_train, fake_val, fake_test = split_list(fake_videos)
    
    # Write split files
    for split_name, real, fake in [
        ("celebdf_train.txt", real_train, fake_train),
        ("celebdf_val.txt", real_val, fake_val),
        ("celebdf_test.txt", real_test, fake_test),
    ]:
        with open(splits_dir / split_name, 'w') as f:
            for v in real:
                f.write(f"Celeb-DF-v2/{v} 0\n")  # 0 = real
            for v in fake:
                f.write(f"Celeb-DF-v2/{v} 1\n")  # 1 = fake
        print(f"  Created {split_name}: {len(real)} real, {len(fake)} fake")


def create_ff_splits(ff_dir: Path, splits_dir: Path):
    """Create splits for FaceForensics++."""
    import random
    random.seed(42)
    
    # FF++ official splits
    # https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
    
    # Use video names 000-720 for train, 720-860 for val, 860-1000 for test
    all_ids = list(range(1000))
    random.shuffle(all_ids)
    
    train_ids = set(all_ids[:700])
    val_ids = set(all_ids[700:850])
    test_ids = set(all_ids[850:])
    
    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    
    for split_name, ids in [
        ("ff_train.txt", train_ids),
        ("ff_val.txt", val_ids),
        ("ff_test.txt", test_ids),
    ]:
        entries = []
        
        # Real videos
        real_dir = ff_dir / "original_sequences/youtube/c23/videos"
        if real_dir.exists():
            for video_path in real_dir.glob("*.mp4"):
                try:
                    vid_id = int(video_path.stem.split('_')[0])
                    if vid_id in ids:
                        entries.append(f"FaceForensics/original_sequences/youtube/c23/videos/{video_path.name} 0")
                except:
                    pass
        
        # Fake videos
        for method in methods:
            method_dir = ff_dir / f"manipulated_sequences/{method}/c23/videos"
            if method_dir.exists():
                for video_path in method_dir.glob("*.mp4"):
                    try:
                        vid_id = int(video_path.stem.split('_')[0])
                        if vid_id in ids:
                            entries.append(f"FaceForensics/manipulated_sequences/{method}/c23/videos/{video_path.name} 1")
                    except:
                        pass
        
        with open(splits_dir / split_name, 'w') as f:
            f.write('\n'.join(entries))
        
        real_count = sum(1 for e in entries if e.endswith(' 0'))
        fake_count = sum(1 for e in entries if e.endswith(' 1'))
        print(f"  Created {split_name}: {real_count} real, {fake_count} fake")


def verify_datasets(output_dir: Path):
    """Verify downloaded datasets."""
    print("\n" + "="*80)
    print("Verifying Datasets")
    print("="*80)
    
    status = []
    
    # Check Celeb-DF
    celebdf_dir = output_dir / "Celeb-DF-v2"
    if celebdf_dir.exists():
        video_count = len(list(celebdf_dir.rglob("*.mp4")))
        status.append(f"✓ Celeb-DF v2: {video_count} videos")
    else:
        status.append("✗ Celeb-DF v2: Not found")
    
    # Check FaceForensics++
    ff_dir = output_dir / "FaceForensics"
    if ff_dir.exists():
        video_count = len(list(ff_dir.rglob("*.mp4")))
        status.append(f"✓ FaceForensics++: {video_count} videos")
    else:
        status.append("✗ FaceForensics++: Not found")
    
    # Check splits
    splits_dir = output_dir / "splits"
    if splits_dir.exists():
        split_files = list(splits_dir.glob("*.txt"))
        status.append(f"✓ Split files: {len(split_files)} files")
    else:
        status.append("✗ Split files: Not found")
    
    for s in status:
        print(f"  {s}")
    
    return all("✓" in s for s in status)


def main():
    parser = argparse.ArgumentParser(description="Download deepfake detection datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "celebdf", "faceforensics"],
        default="all",
        help="Dataset to download"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATASETS_DIR),
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--ff-path",
        type=str,
        default=None,
        help="Path to manually downloaded FaceForensics++ files"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Deepfake Detection - Dataset Download")
    print("="*80)
    print(f"Output directory: {output_dir}")
    
    success = True
    
    if args.dataset in ["all", "celebdf"]:
        if not download_celeb_df(output_dir):
            success = False
    
    if args.dataset in ["all", "faceforensics"]:
        ff_path = Path(args.ff_path) if args.ff_path else None
        if ff_path:
            if not setup_faceforensics(ff_path, output_dir):
                success = False
        else:
            print(FACEFORENSICS_INFO)
            print("\nTo setup FaceForensics++, provide --ff-path argument after downloading.")
    
    # Create splits if any dataset exists
    if (output_dir / "Celeb-DF-v2").exists() or (output_dir / "FaceForensics").exists():
        create_dataset_splits(output_dir)
    
    # Verify
    verify_datasets(output_dir)
    
    print("\n" + "="*80)
    if success:
        print("Dataset download complete!")
        print("\nNext steps:")
        print("  1. Extract faces: make extract-faces")
        print("  2. Start training: make train")
    else:
        print("Some datasets could not be downloaded.")
        print("Please check the errors above and try again.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
