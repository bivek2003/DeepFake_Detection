#!/usr/bin/env python
"""
Prepare train/val/test splits for deepfake detection training.
Scans the faces directory and creates split files.
"""

import os
import random
from pathlib import Path
from collections import defaultdict
import argparse


def scan_faces_directory(faces_dir: Path) -> dict:
    """Scan faces directory and categorize by real/fake."""
    samples = {
        'real': [],
        'fake': []
    }
    
    faces_dir = Path(faces_dir)
    
    # Celeb-DF structure
    celeb_df = faces_dir / 'Celeb-DF-v2'
    if celeb_df.exists():
        # Real: Celeb-real, YouTube-real
        for real_dir in ['Celeb-real', 'YouTube-real']:
            real_path = celeb_df / real_dir
            if real_path.exists():
                for img in real_path.glob('**/*.jpg'):
                    samples['real'].append(str(img))
        
        # Fake: Celeb-synthesis
        fake_path = celeb_df / 'Celeb-synthesis'
        if fake_path.exists():
            for img in fake_path.glob('**/*.jpg'):
                samples['fake'].append(str(img))
    
    # FaceForensics++ structure
    ff_dir = faces_dir / 'FaceForensics'
    if ff_dir.exists():
        # Real: original_sequences
        real_path = ff_dir / 'original_sequences'
        if real_path.exists():
            for img in real_path.glob('**/*.jpg'):
                samples['real'].append(str(img))
        
        # Fake: manipulated_sequences
        fake_path = ff_dir / 'manipulated_sequences'
        if fake_path.exists():
            for img in fake_path.glob('**/*.jpg'):
                samples['fake'].append(str(img))
    
    return samples


def create_splits(samples: dict, train_ratio: float = 0.7, val_ratio: float = 0.15) -> dict:
    """Create train/val/test splits maintaining class balance."""
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for label, label_samples in [('real', samples['real']), ('fake', samples['fake'])]:
        # Shuffle
        random.shuffle(label_samples)
        
        n = len(label_samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        label_int = 0 if label == 'real' else 1
        
        for i, img_path in enumerate(label_samples):
            if i < train_end:
                splits['train'].append((img_path, label_int))
            elif i < val_end:
                splits['val'].append((img_path, label_int))
            else:
                splits['test'].append((img_path, label_int))
    
    # Shuffle each split
    for split in splits.values():
        random.shuffle(split)
    
    return splits


def write_splits(splits: dict, output_dir: Path):
    """Write split files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, samples in splits.items():
        split_file = output_dir / f'{split_name}.txt'
        with open(split_file, 'w') as f:
            for img_path, label in samples:
                f.write(f'{img_path} {label}\n')
        print(f'Written {split_name}.txt: {len(samples)} samples')


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset splits')
    parser.add_argument('--faces-dir', type=str, default='datasets/faces',
                       help='Directory containing extracted faces')
    parser.add_argument('--output-dir', type=str, default='datasets/splits',
                       help='Output directory for split files')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print('='*60)
    print('Preparing Dataset Splits')
    print('='*60)
    
    # Scan faces
    print(f'\nScanning faces directory: {args.faces_dir}')
    samples = scan_faces_directory(args.faces_dir)
    
    print(f'\nFound:')
    print(f'  Real faces: {len(samples["real"])}')
    print(f'  Fake faces: {len(samples["fake"])}')
    print(f'  Total: {len(samples["real"]) + len(samples["fake"])}')
    
    if len(samples['real']) == 0 or len(samples['fake']) == 0:
        print('\nERROR: No samples found in one or both categories!')
        return
    
    # Create splits
    print(f'\nCreating splits (train={args.train_ratio}, val={args.val_ratio}, test={1-args.train_ratio-args.val_ratio})')
    splits = create_splits(samples, args.train_ratio, args.val_ratio)
    
    print(f'\nSplit sizes:')
    for split_name, split_samples in splits.items():
        real_count = sum(1 for _, l in split_samples if l == 0)
        fake_count = sum(1 for _, l in split_samples if l == 1)
        print(f'  {split_name}: {len(split_samples)} (real={real_count}, fake={fake_count})')
    
    # Write splits
    print(f'\nWriting splits to: {args.output_dir}')
    write_splits(splits, args.output_dir)
    
    print('\n' + '='*60)
    print('Split preparation complete!')
    print('='*60)


if __name__ == '__main__':
    main()
