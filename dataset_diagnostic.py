#!/usr/bin/env python3
"""
Diagnose your dataset structure to find all videos
"""

from pathlib import Path
from collections import defaultdict
import re

def analyze_dataset(data_root='datasets'):
    data_root = Path(data_root)
    
    print("="*80)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*80)
    
    # Find all MP4 files
    all_videos = list(data_root.rglob('*.mp4'))
    print(f"\n📊 Total MP4 files found: {len(all_videos):,}")
    
    # Group by directory structure
    dir_groups = defaultdict(list)
    for video in all_videos:
        # Get relative path from data_root
        rel_path = video.relative_to(data_root)
        dir_key = str(rel_path.parent)
        dir_groups[dir_key].append(video)
    
    print(f"\n📁 Found {len(dir_groups)} unique directory structures:")
    print("\n" + "="*80)
    
    for dir_path in sorted(dir_groups.keys()):
        videos = dir_groups[dir_path]
        print(f"\n📂 {dir_path}/")
        print(f"   Videos: {len(videos):,}")
        
        # Sample filenames to understand naming pattern
        sample_names = [v.name for v in videos[:5]]
        print(f"   Samples: {sample_names[:3]}")
        
        # Check naming patterns
        actor_pattern = 0
        pair_pattern = 0
        other_pattern = 0
        
        for video in videos:
            if re.match(r'\d+__', video.stem):
                actor_pattern += 1
            elif re.match(r'\d+_\d+', video.stem):
                pair_pattern += 1
            else:
                other_pattern += 1
        
        print(f"   Patterns:")
        if actor_pattern > 0:
            print(f"     - Actor format (XX__scene): {actor_pattern}")
        if pair_pattern > 0:
            print(f"     - Pair format (XXX_YYY): {pair_pattern}")
        if other_pattern > 0:
            print(f"     - Other format: {other_pattern}")
    
    # Analyze by top-level folders
    print("\n" + "="*80)
    print("SUMMARY BY TOP-LEVEL FOLDER")
    print("="*80)
    
    top_level = defaultdict(int)
    for video in all_videos:
        rel_path = video.relative_to(data_root)
        top_folder = rel_path.parts[0]
        top_level[top_folder] += 1
    
    for folder, count in sorted(top_level.items()):
        print(f"   {folder:30s}: {count:6,} videos")
    
    # Check for actor IDs
    print("\n" + "="*80)
    print("ACTOR ID ANALYSIS")
    print("="*80)
    
    actor_ids = set()
    for video in all_videos:
        match = re.match(r'(\d+)__', video.stem)
        if match:
            actor_ids.add(match.group(1))
        match = re.match(r'(\d+)_(\d+)', video.stem)
        if match:
            actor_ids.add(match.group(1))
            actor_ids.add(match.group(2))
    
    print(f"   Unique actor IDs found: {len(actor_ids)}")
    print(f"   Sample IDs: {sorted(list(actor_ids))[:10]}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if len(all_videos) < 3000:
        print("\n⚠️  WARNING: Only found", len(all_videos), "videos")
        print("   Expected: 10,000+ videos")
        print("\n   Possible issues:")
        print("   1. Videos are compressed/archived")
        print("   2. Videos are in unexpected folders")
        print("   3. Dataset is incomplete")
    else:
        print("\n✅ Found sufficient videos!")
        print(f"   Total: {len(all_videos):,} videos")
    
    # Generate path configuration
    print("\n📝 Detected paths for script configuration:")
    print("\nOriginal video paths:")
    for dir_path, videos in dir_groups.items():
        if 'original' in dir_path.lower() or any(re.match(r'\d+__', v.stem) for v in videos[:10]):
            print(f"   - {dir_path}/")
    
    print("\nManipulated video paths:")
    for dir_path, videos in dir_groups.items():
        if 'manipulated' in dir_path.lower() or any(re.match(r'\d+_\d+', v.stem) for v in videos[:10]):
            print(f"   - {dir_path}/")
    
    print("\nInternet video paths:")
    for dir_path, videos in dir_groups.items():
        if 'internet' in dir_path.lower():
            print(f"   - {dir_path}/")


if __name__ == '__main__':
    analyze_dataset('datasets')
