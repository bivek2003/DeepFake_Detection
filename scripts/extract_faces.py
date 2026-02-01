#!/usr/bin/env python3
"""
Face Extraction Script for Deepfake Detection Training

Extracts faces from video datasets and saves them as images.

Usage:
    python scripts/extract_faces.py --dataset all
    python scripts/extract_faces.py --dataset celebdf --num-frames 32
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import argparse
import torch
from app.ml.data.face_extractor import FaceExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract faces from video datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "celebdf", "faceforensics"],
        default="all",
        help="Dataset to process"
    )
    parser.add_argument(
        "--datasets-dir",
        type=str,
        default="/app/datasets" if Path("/app/datasets").exists() else "./datasets",
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for faces (default: {datasets-dir}/faces)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames to extract per video"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=380,
        help="Output face image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for face detection"
    )
    
    args = parser.parse_args()
    
    datasets_dir = Path(args.datasets_dir)
    output_dir = Path(args.output_dir) if args.output_dir else datasets_dir / "faces"
    
    print("="*80)
    print("Face Extraction for Deepfake Detection")
    print("="*80)
    print(f"Datasets directory: {datasets_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Frames per video: {args.num_frames}")
    print(f"Face size: {args.size}x{args.size}")
    print("="*80)
    
    extractor = FaceExtractor(
        output_size=args.size,
        device=args.device,
    )
    
    total_results = {
        "total": 0,
        "success": 0,
        "skipped": 0,
        "failed": 0,
        "total_faces": 0,
    }
    
    # Process Celeb-DF
    if args.dataset in ["all", "celebdf"]:
        celebdf_dir = datasets_dir / "Celeb-DF-v2"
        if celebdf_dir.exists():
            print("\n" + "-"*40)
            print("Processing Celeb-DF v2...")
            print("-"*40)
            
            results = extractor.process_dataset(
                str(celebdf_dir),
                str(output_dir / "Celeb-DF-v2"),
                num_frames=args.num_frames,
            )
            
            for key in total_results:
                total_results[key] += results.get(key, 0)
            
            print(f"Celeb-DF complete: {results['total_faces']} faces extracted")
        else:
            print(f"\nCeleb-DF not found at {celebdf_dir}")
    
    # Process FaceForensics++
    if args.dataset in ["all", "faceforensics"]:
        ff_dir = datasets_dir / "FaceForensics"
        if ff_dir.exists():
            print("\n" + "-"*40)
            print("Processing FaceForensics++...")
            print("-"*40)
            
            # Process original sequences
            original_dir = ff_dir / "original_sequences" / "youtube" / "c23" / "videos"
            if original_dir.exists():
                print("\nProcessing original sequences...")
                results = extractor.process_dataset(
                    str(original_dir),
                    str(output_dir / "FaceForensics" / "original"),
                    num_frames=args.num_frames,
                )
                for key in total_results:
                    total_results[key] += results.get(key, 0)
            
            # Process manipulated sequences
            methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
            for method in methods:
                method_dir = ff_dir / "manipulated_sequences" / method / "c23" / "videos"
                if method_dir.exists():
                    print(f"\nProcessing {method}...")
                    results = extractor.process_dataset(
                        str(method_dir),
                        str(output_dir / "FaceForensics" / method),
                        num_frames=args.num_frames,
                    )
                    for key in total_results:
                        total_results[key] += results.get(key, 0)
            
            print(f"FaceForensics++ complete")
        else:
            print(f"\nFaceForensics++ not found at {ff_dir}")
    
    # Print summary
    print("\n" + "="*80)
    print("Extraction Summary")
    print("="*80)
    print(f"Total videos processed: {total_results['total']}")
    print(f"  Successful: {total_results['success']}")
    print(f"  Skipped (already exists): {total_results['skipped']}")
    print(f"  Failed: {total_results['failed']}")
    print(f"Total faces extracted: {total_results['total_faces']}")
    print("="*80)
    
    # Update split files with face paths
    print("\nUpdating split files for face images...")
    update_splits_for_faces(datasets_dir, output_dir)
    
    print("\nDone! Next step: make train")


def update_splits_for_faces(datasets_dir: Path, faces_dir: Path):
    """Update split files to point to extracted face images."""
    splits_dir = datasets_dir / "splits"
    
    if not splits_dir.exists():
        print("No splits directory found, skipping split update.")
        return
    
    for split_file in splits_dir.glob("*.txt"):
        updated_lines = []
        
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                video_path = parts[0]
                label = parts[1]
                
                # Convert video path to faces directory path
                # e.g., Celeb-DF-v2/Celeb-real/id0_0000.mp4 -> faces/Celeb-DF-v2/Celeb-real/id0_0000/
                video_stem = Path(video_path).stem
                parent_path = Path(video_path).parent
                faces_path = faces_dir / parent_path / video_stem
                
                if faces_path.exists():
                    updated_lines.append(f"{faces_path} {label}")
                else:
                    # Try alternative path structure
                    alt_path = faces_dir / video_path.replace('.mp4', '').replace('.avi', '')
                    if alt_path.exists():
                        updated_lines.append(f"{alt_path} {label}")
        
        # Write updated split file for faces
        faces_split_file = splits_dir / f"{split_file.stem}_faces.txt"
        with open(faces_split_file, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        print(f"  Created {faces_split_file.name} with {len(updated_lines)} entries")


if __name__ == "__main__":
    main()
