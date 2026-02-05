#!/usr/bin/env python
"""
FaceForensics++ Download Helper

This script automates downloading FaceForensics++ dataset.
Requires the official download script from FaceForensics++ repository.

Usage:
    1. Save the official FF++ download script as 'ff_download.py' in the same directory
    2. Run: python download_faceforensics.py --output-dir /path/to/datasets
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


# Datasets to download for deepfake detection
DATASETS = [
    ('original', 'Real YouTube videos'),
    ('Deepfakes', 'Deepfakes manipulation'),
    ('Face2Face', 'Face2Face manipulation'),
    ('FaceSwap', 'FaceSwap manipulation'),
    ('NeuralTextures', 'NeuralTextures manipulation'),
]

# Compression level (c23 is recommended - good quality, reasonable size)
COMPRESSION = 'c23'


def check_download_script(script_path: Path) -> bool:
    """Check if the official download script exists."""
    return script_path.exists()


def download_dataset(
    script_path: Path,
    output_dir: Path,
    dataset: str,
    compression: str = 'c23',
    server: str = 'EU'
):
    """Download a single dataset."""
    cmd = [
        sys.executable,
        str(script_path),
        str(output_dir),
        '-d', dataset,
        '-c', compression,
        '-t', 'videos',
        '--server', server
    ]
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # Run interactively so user can accept TOS
    result = subprocess.run(cmd)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Download FaceForensics++ datasets for deepfake detection'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='.',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--script', '-s',
        type=str,
        default='ff_download.py',
        help='Path to official FaceForensics++ download script'
    )
    parser.add_argument(
        '--compression', '-c',
        type=str,
        default='c23',
        choices=['raw', 'c23', 'c40'],
        help='Compression level (c23 recommended)'
    )
    parser.add_argument(
        '--server',
        type=str,
        default='EU',
        choices=['EU', 'EU2', 'CA'],
        help='Download server (try different if slow)'
    )
    parser.add_argument(
        '--datasets', '-d',
        nargs='+',
        default=None,
        help='Specific datasets to download (default: all)'
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    script_path = Path(args.script).resolve()
    
    # Check for download script
    if not check_download_script(script_path):
        print(f"ERROR: FaceForensics++ download script not found at: {script_path}")
        print("\nPlease save the official download script from:")
        print("https://github.com/ondyari/FaceForensics")
        print(f"\nSave it as: {script_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FaceForensics++ Dataset Download")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Compression: {args.compression}")
    print(f"Server: {args.server}")
    print()
    
    # Determine which datasets to download
    if args.datasets:
        datasets_to_download = [(d, d) for d in args.datasets]
    else:
        datasets_to_download = DATASETS
    
    print("Datasets to download:")
    for dataset, description in datasets_to_download:
        print(f"  - {dataset}: {description}")
    print()
    
    # Download each dataset
    success_count = 0
    failed = []
    
    for dataset, description in datasets_to_download:
        try:
            success = download_dataset(
                script_path=script_path,
                output_dir=output_dir,
                dataset=dataset,
                compression=args.compression,
                server=args.server
            )
            if success:
                success_count += 1
                print(f"✓ {dataset} downloaded successfully")
            else:
                failed.append(dataset)
                print(f"✗ {dataset} download failed")
        except KeyboardInterrupt:
            print("\n\nDownload interrupted by user.")
            break
        except Exception as e:
            print(f"✗ Error downloading {dataset}: {e}")
            failed.append(dataset)
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"Successfully downloaded: {success_count}/{len(datasets_to_download)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    
    print("\nNext steps:")
    print("1. Run face extraction: make extract-faces")
    print("2. Start training: make train-production")
    print("="*60)


if __name__ == "__main__":
    main()
