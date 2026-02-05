#!/usr/bin/env python3
"""
Create FaceForensics++ split files (ff_train.txt, ff_val.txt, ff_test.txt).

Run after extracting faces from FaceForensics videos:
    python scripts/extract_faces.py --dataset faceforensics
    python scripts/prepare_ff_splits.py

These splits enable training to use both Celeb-DF and FaceForensics datasets.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def main():
    datasets_dir = Path("/app/datasets") if Path("/app/datasets").exists() else Path("./datasets")
    faces_dir = datasets_dir / "faces"
    splits_dir = datasets_dir / "splits"
    ff_faces = faces_dir / "FaceForensics"

    if not ff_faces.exists():
        print(f"FaceForensics faces not found at {ff_faces}")
        print("Run: python scripts/extract_faces.py --dataset faceforensics")
        sys.exit(1)

    real_samples = []
    fake_samples = []

    for real_dir in ["original_sequences", "original"]:
        real_path = ff_faces / real_dir
        if real_path.exists():
            for img_path in real_path.rglob("*.jpg"):
                real_samples.append(str(img_path))
            break

    manipulated = ff_faces / "manipulated_sequences"
    if manipulated.exists():
        for img_path in manipulated.rglob("*.jpg"):
            fake_samples.append(str(img_path))
    else:
        for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            method_dir = ff_faces / method
            if method_dir.exists():
                for img_path in method_dir.rglob("*.jpg"):
                    fake_samples.append(str(img_path))

    print(f"FaceForensics: {len(real_samples)} real, {len(fake_samples)} fake")

    if not real_samples and not fake_samples:
        print("No FaceForensics face images found.")
        sys.exit(1)

    random.seed(42)
    splits = {"train": [], "val": [], "test": []}
    train_ratio, val_ratio = 0.7, 0.15

    def to_rel(p):
        try:
            return str(Path(p).relative_to(datasets_dir))
        except ValueError:
            return p

    for samples, label in [(real_samples, 0), (fake_samples, 1)]:
        if not samples:
            continue
        random.shuffle(samples)
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        for i, path in enumerate(samples):
            entry = (to_rel(path), label)
            if i < train_end:
                splits["train"].append(entry)
            elif i < val_end:
                splits["val"].append(entry)
            else:
                splits["test"].append(entry)

    for split in splits.values():
        random.shuffle(split)

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, entries in splits.items():
        if entries:
            out = splits_dir / f"ff_{name}.txt"
            with open(out, "w") as f:
                for path, label in entries:
                    f.write(f"{path} {label}\n")
            print(f"Created {out.name}: {len(entries)} samples")


if __name__ == "__main__":
    main()
