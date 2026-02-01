#!/usr/bin/env python3
"""
Export trained model for production deployment.

Creates optimized model files:
- PyTorch state dict for inference
- ONNX export for cross-platform deployment
- TorchScript for optimized inference
- Model metadata and configuration

Usage:
    python scripts/export_model.py --checkpoint checkpoints/best_model.pt --output weights/
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import argparse
import json
import shutil
from datetime import datetime
from typing import Dict, Any

import torch
import torch.nn as nn

from app.ml.architecture import DeepfakeDetector, create_model


def export_state_dict(
    model: DeepfakeDetector,
    output_path: str,
    temperature: float = 1.0,
    threshold: float = 0.5,
    metadata: Dict[str, Any] = None,
):
    """
    Export model state dict with metadata.
    
    Args:
        model: Trained model
        output_path: Output file path
        temperature: Calibration temperature
        threshold: Classification threshold
        metadata: Additional metadata
    """
    export_data = {
        "model_state_dict": model.state_dict(),
        "temperature": temperature,
        "threshold": threshold,
        "backbone": model.backbone_name,
        "num_classes": model.num_classes,
        "exported_at": datetime.now().isoformat(),
    }
    
    if metadata:
        export_data["metadata"] = metadata
    
    torch.save(export_data, output_path)
    print(f"Exported state dict: {output_path}")


def export_torchscript(
    model: DeepfakeDetector,
    output_path: str,
    image_size: int = 380,
):
    """
    Export model to TorchScript format.
    
    Args:
        model: Trained model
        output_path: Output file path
        image_size: Input image size
    """
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, image_size, image_size)
    
    if next(model.parameters()).is_cuda:
        example_input = example_input.cuda()
    
    # Trace model
    try:
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(output_path)
        print(f"Exported TorchScript: {output_path}")
    except Exception as e:
        print(f"TorchScript export failed: {e}")
        
        # Try scripting instead
        try:
            scripted_model = torch.jit.script(model)
            scripted_model.save(output_path)
            print(f"Exported TorchScript (scripted): {output_path}")
        except Exception as e2:
            print(f"TorchScript scripting also failed: {e2}")


def export_onnx(
    model: DeepfakeDetector,
    output_path: str,
    image_size: int = 380,
    opset_version: int = 14,
):
    """
    Export model to ONNX format.
    
    Args:
        model: Trained model
        output_path: Output file path
        image_size: Input image size
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Create example input on CPU
    example_input = torch.randn(1, 3, image_size, image_size)
    model_cpu = model.cpu()
    
    try:
        torch.onnx.export(
            model_cpu,
            example_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
        )
        print(f"Exported ONNX: {output_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")


def create_model_card(
    output_dir: str,
    checkpoint_path: str,
    metrics: Dict[str, Any] = None,
    config: Dict[str, Any] = None,
):
    """
    Create model card with documentation.
    
    Args:
        output_dir: Output directory
        checkpoint_path: Original checkpoint path
        metrics: Model metrics
        config: Training configuration
    """
    model_card = {
        "name": "Deepfake Detection Model",
        "version": "1.0.0",
        "description": "EfficientNet-B4 based binary classifier for deepfake detection",
        "architecture": {
            "backbone": config.get("model", {}).get("backbone", "efficientnet_b4") if config else "efficientnet_b4",
            "input_size": 380,
            "output": "Binary classification (0=Real, 1=Fake)",
        },
        "training": {
            "datasets": ["FaceForensics++", "Celeb-DF v2"],
            "epochs": config.get("training", {}).get("epochs", "N/A") if config else "N/A",
            "batch_size": config.get("training", {}).get("batch_size", "N/A") if config else "N/A",
        },
        "performance": metrics or {},
        "usage": {
            "preprocessing": "Resize to 380x380, normalize with ImageNet mean/std",
            "inference": "Pass through model, apply sigmoid to get probability",
            "threshold": "0.5 (default), adjust based on use case",
        },
        "limitations": [
            "Trained on specific manipulation methods - may not generalize to all deepfakes",
            "Performance may vary on compressed or low-quality videos",
            "Requires face detection preprocessing",
        ],
        "ethical_considerations": [
            "This is a forensic tool providing probability estimates, not certainty",
            "Results should be verified by experts before taking action",
            "False positives and false negatives are possible",
        ],
        "exported_at": datetime.now().isoformat(),
        "source_checkpoint": str(checkpoint_path),
    }
    
    card_path = Path(output_dir) / "model_card.json"
    with open(card_path, 'w') as f:
        json.dump(model_card, f, indent=2)
    
    print(f"Created model card: {card_path}")
    
    # Also create markdown version
    md_content = f"""# Deepfake Detection Model

## Overview
- **Architecture**: {model_card['architecture']['backbone']}
- **Input Size**: {model_card['architecture']['input_size']}x{model_card['architecture']['input_size']}
- **Output**: {model_card['architecture']['output']}

## Training
- **Datasets**: {', '.join(model_card['training']['datasets'])}
- **Epochs**: {model_card['training']['epochs']}

## Performance
{json.dumps(model_card['performance'], indent=2) if model_card['performance'] else 'See evaluation results'}

## Usage

### Preprocessing
{model_card['usage']['preprocessing']}

### Inference
{model_card['usage']['inference']}

### Threshold
{model_card['usage']['threshold']}

## Limitations
{chr(10).join(f'- {l}' for l in model_card['limitations'])}

## Ethical Considerations
{chr(10).join(f'- {e}' for e in model_card['ethical_considerations'])}

---
Exported: {model_card['exported_at']}
"""
    
    md_path = Path(output_dir) / "MODEL_CARD.md"
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    print(f"Created model card (MD): {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained model for production")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="./weights", help="Output directory")
    parser.add_argument("--backbone", type=str, default="efficientnet_b4", help="Model backbone")
    parser.add_argument("--formats", type=str, nargs="+", 
                       default=["state_dict", "torchscript"],
                       choices=["state_dict", "torchscript", "onnx"],
                       help="Export formats")
    parser.add_argument("--temperature", type=float, default=1.0, help="Calibration temperature")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Model Export")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Formats: {args.formats}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get configuration
    config = checkpoint.get("config", {})
    metrics = checkpoint.get("metrics", {})
    
    # Determine backbone
    backbone = config.get("model", {}).get("backbone", args.backbone)
    
    # Create model
    model = DeepfakeDetector(
        backbone=backbone,
        pretrained=False,
    )
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Get calibration parameters
    temperature = checkpoint.get("temperature", args.temperature)
    threshold = checkpoint.get("threshold", args.threshold)
    
    print(f"\nModel loaded: {backbone}")
    print(f"Temperature: {temperature}")
    print(f"Threshold: {threshold}")
    
    # Export formats
    print("\nExporting...")
    
    if "state_dict" in args.formats:
        export_state_dict(
            model,
            str(output_dir / "deepfake_detector.pt"),
            temperature=temperature,
            threshold=threshold,
            metadata={"backbone": backbone, "metrics": metrics},
        )
    
    if "torchscript" in args.formats:
        export_torchscript(
            model,
            str(output_dir / "deepfake_detector_traced.pt"),
        )
    
    if "onnx" in args.formats:
        export_onnx(
            model,
            str(output_dir / "deepfake_detector.onnx"),
        )
    
    # Create model card
    create_model_card(
        str(output_dir),
        args.checkpoint,
        metrics=metrics,
        config=config,
    )
    
    # Copy checkpoint to weights directory
    shutil.copy(checkpoint_path, output_dir / "checkpoint.pt")
    
    print("\n" + "="*80)
    print("Export complete!")
    print(f"Files saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
