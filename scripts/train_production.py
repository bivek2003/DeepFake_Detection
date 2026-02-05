"""
Production Training Script for Maximum Accuracy Deepfake Detection

Features:
- Ensemble model (EfficientNet-B4 + B7 + XceptionNet)
- Multi-dataset training (Celeb-DF + FaceForensics++ + DFDC)
- Advanced augmentation pipeline
- Mixed precision training (FP16)
- Gradient accumulation for large effective batch size
- Learning rate scheduling with warmup
- Early stopping with patience
- Test-time augmentation (TTA)
- Model checkpointing with best AUC tracking

Target: 96%+ accuracy
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.ml.ensemble_architecture import (
    DeepfakeEnsemble, EfficientNetDetector, XceptionNet,
    CombinedLoss, create_production_model
)
from app.ml.data.transforms import get_train_transforms, get_val_transforms


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class MultiDatasetLoader(Dataset):
    """
    Unified dataset loader for multiple deepfake datasets.
    
    Supports:
    - Celeb-DF v2
    - FaceForensics++ (all manipulation types)
    - DFDC Preview
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform=None,
        datasets: List[str] = ["celeb-df", "faceforensics", "dfdc"],
        balance_classes: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.balance_classes = balance_classes
        
        self.samples = []  # List of (image_path, label, dataset_name)
        self.class_counts = {0: 0, 1: 0}
        
        # Load each dataset
        for dataset_name in datasets:
            if dataset_name == "celeb-df":
                self._load_celeb_df()
            elif dataset_name == "faceforensics":
                self._load_faceforensics()
            elif dataset_name == "dfdc":
                self._load_dfdc()
                
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
        logger.info(f"Class distribution: Real={self.class_counts[0]}, Fake={self.class_counts[1]}")
        
    def _load_celeb_df(self):
        """Load Celeb-DF v2 dataset from split files or directory structure."""
        # Try to load from unified split file first
        split_file = self.data_root / "splits" / f"{self.split}.txt"
        
        if split_file.exists():
            logger.info(f"Loading from split file: {split_file}")
            with open(split_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        label = int(parts[1])
                        if Path(img_path).exists():
                            self.samples.append((img_path, label, "celeb-df"))
                            self.class_counts[label] += 1
            return
            
        # Fall back to directory structure
        faces_dir = self.data_root / "faces" / "Celeb-DF-v2"
        
        if not faces_dir.exists():
            logger.warning(f"Celeb-DF faces not found at {faces_dir}")
            return
            
        # Auto-detect from directory structure
        for category in ["Celeb-real", "YouTube-real"]:
            cat_dir = faces_dir / category
            if cat_dir.exists():
                for img in cat_dir.glob("**/*.jpg"):
                    self.samples.append((str(img), 0, "celeb-df"))
                    self.class_counts[0] += 1
                    
        for category in ["Celeb-synthesis"]:
            cat_dir = faces_dir / category
            if cat_dir.exists():
                for img in cat_dir.glob("**/*.jpg"):
                    self.samples.append((str(img), 1, "celeb-df"))
                    self.class_counts[1] += 1
                        
    def _load_faceforensics(self):
        """Load FaceForensics++ dataset (raw videos path or extracted faces)."""
        ff_dir = self.data_root / "faces" / "FaceForensics"
        
        if not ff_dir.exists():
            ff_dir = self.data_root / "FaceForensics"
            
        if not ff_dir.exists():
            logger.warning(f"FaceForensics++ not found")
            return
        
        # Extracted faces structure (from extract_faces.py): original/, Deepfakes/, Face2Face/, etc.
        real_dir = ff_dir / "original"
        if real_dir.exists():
            for img in real_dir.glob("**/*.jpg"):
                self.samples.append((str(img), 0, "faceforensics"))
                self.class_counts[0] += 1
        
        # Raw videos path (original_sequences/youtube/c23/videos has mp4, not jpg - skip)
        # Alternative: original_sequences/youtube/c23 with extracted faces
        if self.class_counts[0] == 0:
            real_dir = ff_dir / "original_sequences" / "youtube" / "c23"
            if real_dir.exists():
                for img in real_dir.glob("**/*.jpg"):
                    self.samples.append((str(img), 0, "faceforensics"))
                    self.class_counts[0] += 1
                
        # Manipulated: extracted faces in Deepfakes/, Face2Face/, FaceSwap/, NeuralTextures/
        manip_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
        for manip in manip_types:
            manip_dir = ff_dir / manip
            if manip_dir.exists():
                for img in manip_dir.glob("**/*.jpg"):
                    self.samples.append((str(img), 1, f"ff-{manip.lower()}"))
                    self.class_counts[1] += 1
            else:
                # Raw path: manipulated_sequences/Deepfakes/c23/
                manip_dir = ff_dir / "manipulated_sequences" / manip / "c23"
                if manip_dir.exists():
                    for img in manip_dir.glob("**/*.jpg"):
                        self.samples.append((str(img), 1, f"ff-{manip.lower()}"))
                        self.class_counts[1] += 1
                    
    def _load_dfdc(self):
        """Load DFDC Preview dataset."""
        dfdc_dir = self.data_root / "faces" / "DFDC"
        
        if not dfdc_dir.exists():
            dfdc_dir = self.data_root / "DFDC"
            
        if not dfdc_dir.exists():
            logger.warning("DFDC not found")
            return
            
        # Load from metadata if available
        metadata_file = dfdc_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
            for video_name, info in metadata.items():
                label = 1 if info.get("label") == "FAKE" else 0
                video_faces = dfdc_dir / video_name.replace(".mp4", "")
                if video_faces.exists():
                    for img in video_faces.glob("*.jpg"):
                        self.samples.append((str(img), label, "dfdc"))
                        self.class_counts[label] += 1
        else:
            # Assume directory structure: real/ and fake/
            for img in (dfdc_dir / "real").glob("**/*.jpg"):
                self.samples.append((str(img), 0, "dfdc"))
                self.class_counts[0] += 1
            for img in (dfdc_dir / "fake").glob("**/*.jpg"):
                self.samples.append((str(img), 1, "dfdc"))
                self.class_counts[1] += 1
                
    def get_sampler(self) -> WeightedRandomSampler:
        """Get weighted sampler for balanced training."""
        weights = []
        total = sum(self.class_counts.values())
        class_weights = {
            cls: total / (2 * count) 
            for cls, count in self.class_counts.items()
        }
        
        for _, label, _ in self.samples:
            weights.append(class_weights[label])
            
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, label, dataset = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return random valid sample
            return self.__getitem__(np.random.randint(len(self)))
            
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        label = torch.tensor([label], dtype=torch.float32)
        
        return image, label, dataset


class ProductionTrainer:
    """
    Production trainer with all optimizations for maximum accuracy.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Directories
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_auc = 0.0
        self.best_acc = 0.0
        self.patience_counter = 0
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_training()
        
        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(self.log_dir / f"run_{timestamp}")
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _setup_model(self):
        """Initialize model."""
        model_type = self.config.get("model_type", "ensemble")
        
        logger.info(f"Initializing {model_type} model...")
        self.model = create_production_model(
            model_type=model_type,
            pretrained=True,
            device=self.device
        )
        
        # Multi-GPU if available
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            
    def _setup_data(self):
        """Initialize data loaders."""
        data_root = self.config.get("data_root", "datasets")
        batch_size = self.config.get("batch_size", 16)
        num_workers = self.config.get("num_workers", 4)
        datasets = self.config.get("datasets", ["celeb-df"])
        
        # Transforms
        train_transform = get_train_transforms(
            image_size=self.config.get("image_size", 380)
        )
        val_transform = get_val_transforms(
            image_size=self.config.get("image_size", 380)
        )
        
        # Datasets
        self.train_dataset = MultiDatasetLoader(
            data_root=data_root,
            split="train",
            transform=train_transform,
            datasets=datasets
        )
        
        self.val_dataset = MultiDatasetLoader(
            data_root=data_root,
            split="val", 
            transform=val_transform,
            datasets=datasets
        )
        
        # Data loaders with balanced sampling
        sampler = self.train_dataset.get_sampler()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        
    def _setup_training(self):
        """Initialize optimizer, scheduler, loss."""
        # Loss function
        self.criterion = CombinedLoss(
            bce_weight=0.5,
            focal_weight=0.5,
            label_smoothing=self.config.get("label_smoothing", 0.1)
        )
        
        # Optimizer with different LR for backbone vs head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if "classifier" in name or "fusion" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": self.config.get("backbone_lr", 1e-5)},
            {"params": head_params, "lr": self.config.get("head_lr", 1e-4)}
        ], weight_decay=self.config.get("weight_decay", 0.01))
        
        # Learning rate scheduler with warmup
        total_steps = len(self.train_loader) * self.config.get("epochs", 100)
        warmup_steps = int(total_steps * self.config.get("warmup_ratio", 0.1))
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
            
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Gradient accumulation
        self.accum_steps = self.config.get("gradient_accumulation", 4)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) / self.accum_steps
                
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accum_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
            total_loss += loss.item() * self.accum_steps
            
            # Collect predictions
            with torch.no_grad():
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * self.accum_steps:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            "loss": total_loss / len(self.train_loader),
            "accuracy": accuracy_score(all_labels, all_preds > 0.5),
            "auc": roc_auc_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds > 0.5)
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, use_tta: bool = False) -> Dict[str, float]:
        """Validate model with optional test-time augmentation."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels, _ in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Use mixed precision for validation too
            with autocast():
                if use_tta:
                    # Test-time augmentation: original + horizontal flip
                    outputs1 = self.model(images)
                    outputs2 = self.model(torch.flip(images, dims=[3]))
                    outputs = (outputs1 + outputs2) / 2
                else:
                    outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Find optimal threshold
        best_thresh = 0.5
        best_f1 = 0
        for thresh in np.arange(0.3, 0.7, 0.05):
            f1 = f1_score(all_labels, all_preds > thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        metrics = {
            "loss": total_loss / len(self.val_loader),
            "accuracy": accuracy_score(all_labels, all_preds > best_thresh),
            "auc": roc_auc_score(all_labels, all_preds),
            "auc_pr": average_precision_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds > best_thresh),
            "recall": recall_score(all_labels, all_preds > best_thresh),
            "f1": best_f1,
            "threshold": best_thresh
        }
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "best_auc": self.best_auc,
            "best_acc": self.best_acc
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            logger.info(f"New best model saved! AUC: {metrics['auc']:.4f}")
            
        # Save periodic checkpoints
        if (self.epoch + 1) % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{self.epoch + 1}.pt")
            
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_auc = checkpoint.get("best_auc", 0.0)
        self.best_acc = checkpoint.get("best_acc", 0.0)
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
        
    def train(self):
        """Full training loop."""
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("patience", 15)
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Early stopping patience: {patience}")
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate(use_tta=epoch >= epochs // 2)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            # TensorBoard logging
            for name, value in train_metrics.items():
                self.writer.add_scalar(f"train/{name}", value, epoch)
            for name, value in val_metrics.items():
                self.writer.add_scalar(f"val/{name}", value, epoch)
                
            # Check for improvement
            is_best = val_metrics["auc"] > self.best_auc
            if is_best:
                self.best_auc = val_metrics["auc"]
                self.best_acc = val_metrics["accuracy"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
        # Final evaluation with TTA
        logger.info("Final evaluation with test-time augmentation...")
        final_metrics = self.validate(use_tta=True)
        logger.info(f"Final AUC: {final_metrics['auc']:.4f}")
        logger.info(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        
        self.writer.close()
        
        return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Production Deepfake Detection Training")
    parser.add_argument("--config", type=str, default="configs/production_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--data-root", type=str, default="datasets")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", type=str, default="ensemble", 
                       choices=["ensemble", "efficientnet_b4", "efficientnet_b7", "xception"])
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        
    # Override with CLI args
    config["data_root"] = args.data_root
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["model_type"] = args.model
    
    # Set defaults for production
    config.setdefault("image_size", 380)
    config.setdefault("backbone_lr", 1e-5)
    config.setdefault("head_lr", 1e-4)
    config.setdefault("weight_decay", 0.01)
    config.setdefault("warmup_ratio", 0.1)
    config.setdefault("gradient_accumulation", 4)
    config.setdefault("label_smoothing", 0.1)
    config.setdefault("patience", 15)
    config.setdefault("num_workers", 4)
    config.setdefault("datasets", ["celeb-df", "faceforensics", "dfdc"])
    
    # Initialize trainer
    trainer = ProductionTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # Train
    final_metrics = trainer.train()
    
    # Save final config
    with open(trainer.checkpoint_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
        
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best AUC: {trainer.best_auc:.4f}")
    print(f"Best Accuracy: {trainer.best_acc:.4f}")
    print(f"Checkpoint saved to: {trainer.checkpoint_dir / 'best.pt'}")
    print("="*60)


if __name__ == "__main__":
    main()
