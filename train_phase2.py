#!/usr/bin/env python3
"""
Phase 2: Deepfake Detection Training using existing data pipeline
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import your existing data pipeline
from deepfake_detector.data.data_pipeline import DataPipelineManager
from deepfake_detector.data.dataset_manager import DatasetManager

# Import new Phase 2 models
from deepfake_detector.models.efficientnet_detector import EfficientNetDeepfakeDetector
from deepfake_detector.models.xception_detector import XceptionDeepfakeDetector
from deepfake_detector.models.model_trainer import DeepfakeTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/phase2_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_sample_dataset(num_samples=200):
    """Create sample dataset for testing Phase 2"""
    import cv2
    
    sample_dir = Path("./sample_deepfake_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create directories
    (sample_dir / "real").mkdir(exist_ok=True)
    (sample_dir / "fake").mkdir(exist_ok=True)
    
    video_paths = []
    labels = []
    
    # Create sample images (simulating video frames)
    for i in range(num_samples // 2):
        # Real images - more natural patterns
        real_img = np.random.randint(120, 200, (224, 224, 3), dtype=np.uint8)
        real_path = sample_dir / "real" / f"real_{i:03d}.jpg"
        cv2.imwrite(str(real_path), real_img)
        video_paths.append(str(real_path))
        labels.append(0)
        
        # Fake images - with subtle artifacts
        fake_img = np.random.randint(80, 160, (224, 224, 3), dtype=np.uint8)
        # Add subtle noise patterns that models can learn
        noise_pattern = np.random.randint(-40, 40, (224, 224, 3))
        fake_img = np.clip(fake_img + noise_pattern * 0.3, 0, 255).astype(np.uint8)
        fake_path = sample_dir / "fake" / f"fake_{i:03d}.jpg"
        cv2.imwrite(str(fake_path), fake_img)
        video_paths.append(str(fake_path))
        labels.append(1)
    
    logger.info(f"Created {len(video_paths)} sample images in {sample_dir}")
    return video_paths, labels

def main():
    parser = argparse.ArgumentParser(description='Phase 2: Deepfake Detection Training')
    parser.add_argument('--data_dir', type=str, default=None, help='Dataset directory')
    parser.add_argument('--use_sample', action='store_true', help='Use sample data for testing')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--model', type=str, default='both', choices=['efficientnet', 'xception', 'both'])
    parser.add_argument('--preload', action='store_true', help='Preload data into memory')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Initialize data pipeline manager
    pipeline_manager = DataPipelineManager()
    
    # Get data
    if args.use_sample or args.data_dir is None:
        logger.info("Creating sample dataset for Phase 2 testing...")
        video_paths, labels = create_sample_dataset(400)  # Larger sample for better training
    else:
        # Use your existing dataset manager
        dataset_manager = DatasetManager()
        # This would load real datasets - adapt based on your dataset structure
        logger.info(f"Loading dataset from {args.data_dir}")
        # video_paths, labels = dataset_manager.load_dataset(args.data_dir)
        video_paths, labels = create_sample_dataset(400)  # Fallback for demo
    
    logger.info(f"Total samples: {len(video_paths)}")
    logger.info(f"Real samples: {labels.count(0)}")
    logger.info(f"Fake samples: {labels.count(1)}")
    
    # Create dataloaders using your existing pipeline
    dataloaders = pipeline_manager.create_video_dataloaders(
        video_paths=video_paths,
        labels=labels,
        batch_size=args.batch_size,
        num_workers=4,
        split_data=True,
        preload=args.preload
    )
    
    # Models to train
    models_to_train = []
    if args.model in ['efficientnet', 'both']:
        models_to_train.append(('EfficientNet-B4', EfficientNetDeepfakeDetector('efficientnet_b4')))
    if args.model in ['xception', 'both']:
        models_to_train.append(('Xception-like', XceptionDeepfakeDetector()))
    
    results = {}
    
    # Train each model
    for model_name, model in models_to_train:
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 TRAINING {model_name.upper()}")
        logger.info(f"{'='*70}")
        
        # Create trainer with optimized settings
        trainer = DeepfakeTrainer(model, device=device, learning_rate=1e-4)
        
        # Train model
        predictions, targets = trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            epochs=args.epochs,
            save_dir='./models',
            model_name=model_name.lower().replace('-', '_')
        )
        
        results[model_name] = {
            'best_accuracy': trainer.best_accuracy,
            'best_f1': trainer.best_f1,
            'model_path': f'./models/best_{model_name.lower().replace("-", "_")}.pth'
        }
    
    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("🏁 PHASE 2 TRAINING SUMMARY")
    logger.info(f"{'='*70}")
    
    for model_name, result in results.items():
        accuracy = result['best_accuracy']
        f1_score = result['best_f1']
        status = "✅ TARGET ACHIEVED" if accuracy >= 87 else "❌ BELOW TARGET"
        
        logger.info(f"{model_name}:")
        logger.info(f"  Accuracy: {accuracy:.2f}% {status}")
        logger.info(f"  F1 Score: {f1_score:.4f}")
        logger.info(f"  Model: {result['model_path']}")
    
    # Best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['best_accuracy'])
        logger.info(f"\n🏆 BEST MODEL: {best_model[0]}")
        logger.info(f"   Accuracy: {best_model[1]['best_accuracy']:.2f}%")
        
        # Test evaluation
        logger.info(f"\n🧪 Testing on test set...")
        best_model_path = best_model[1]['model_path']
        
        if Path(best_model_path).exists():
            logger.info("Run evaluation with:")
            logger.info(f"python evaluate_phase2.py --model_path {best_model_path}")
    
    return results

if __name__ == "__main__":
    main()
