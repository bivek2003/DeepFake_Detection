#!/usr/bin/env python3
"""
Training Script for Deepfake Detection Model

Features:
- Mixed precision training (FP16) for memory efficiency
- Gradient accumulation for effective larger batch sizes
- Learning rate scheduling with warmup
- Early stopping based on validation AUC
- TensorBoard logging
- Checkpoint saving and resumption

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --epochs 30 --batch-size 32
"""

import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import argparse
import yaml
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from app.ml.architecture import DeepfakeDetector, CombinedLoss, FocalLoss
from app.ml.data.datasets import create_dataloaders, CombinedDataset, DeepfakeDataset
from app.ml.data.transforms import get_train_transforms, get_val_transforms


# Default training configuration (target: >90% F1/AUC/accuracy)
DEFAULT_CONFIG = {
    "model": {
        "backbone": "efficientnet_b4",
        "pretrained": True,
        "dropout": 0.4,
        "hidden_size": 512,
    },
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 8e-5,
        "weight_decay": 2e-5,
        "warmup_epochs": 3,
        "grad_accumulation": 2,
        "mixed_precision": True,
        "early_stop_patience": 10,
        "min_delta": 0.0005,
    },
    "data": {
        "datasets_dir": "/app/datasets",
        "image_size": 380,
        "num_workers": 4,
        "balance_classes": True,
    },
    "loss": {
        "type": "combined",  # 'bce', 'focal', 'combined'
        "bce_weight": 0.5,
        "focal_weight": 0.5,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
    },
    "checkpoints": {
        "save_dir": "/app/checkpoints",
        "save_every": 1,
        "keep_best": 3,
    },
    "logging": {
        "log_dir": "/app/logs",
        "log_every": 50,
    },
}


class Trainer:
    """
    Trainer class for deepfake detection model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Create directories
        self.checkpoint_dir = Path(config["checkpoints"]["save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config["logging"]["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_model()
        self._init_dataloaders()
        self._init_optimizer()
        self._init_loss()
        self._init_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # Mixed precision
        self.use_amp = config["training"]["mixed_precision"]
        self.scaler = GradScaler("cuda") if self.use_amp else None
    
    def _init_model(self):
        """Initialize model."""
        model_config = self.config["model"]
        
        self.model = DeepfakeDetector(
            backbone=model_config["backbone"],
            pretrained=model_config["pretrained"],
            dropout_rate=model_config["dropout"],
            hidden_size=model_config["hidden_size"],
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model: {model_config['backbone']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def _init_dataloaders(self):
        """Initialize data loaders."""
        data_config = self.config["data"]
        training_config = self.config["training"]
        
        train_transform = get_train_transforms(data_config["image_size"])
        val_transform = get_val_transforms(data_config["image_size"])
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            root_dir=data_config["datasets_dir"],
            batch_size=training_config["batch_size"],
            num_workers=data_config["num_workers"],
            train_transform=train_transform,
            val_transform=val_transform,
            balance_training=data_config["balance_classes"],
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        training_config = self.config["training"]
        
        # Different learning rates for backbone and head
        backbone_params = list(self.model.backbone.parameters())
        head_params = list(self.model.classifier.parameters())
        
        param_groups = [
            {"params": backbone_params, "lr": training_config["learning_rate"] * 0.1},
            {"params": head_params, "lr": training_config["learning_rate"]},
        ]
        
        self.optimizer = AdamW(
            param_groups,
            lr=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
        )
        
        # Learning rate scheduler
        steps_per_epoch = len(self.train_loader) // training_config["grad_accumulation"]
        total_steps = steps_per_epoch * training_config["epochs"]
        warmup_steps = steps_per_epoch * training_config["warmup_epochs"]
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[training_config["learning_rate"] * 0.1, training_config["learning_rate"]],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
        )
    
    def _init_loss(self):
        """Initialize loss function."""
        loss_config = self.config["loss"]
        
        if loss_config["type"] == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_config["type"] == "focal":
            self.criterion = FocalLoss(
                alpha=loss_config["focal_alpha"],
                gamma=loss_config["focal_gamma"],
            )
        else:  # combined
            self.criterion = CombinedLoss(
                bce_weight=loss_config["bce_weight"],
                focal_weight=loss_config["focal_weight"],
                focal_alpha=loss_config["focal_alpha"],
                focal_gamma=loss_config["focal_gamma"],
            )
    
    def _init_logging(self):
        """Initialize logging."""
        if TENSORBOARD_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir / f"run_{timestamp}")
            )
        else:
            self.writer = None
            print("TensorBoard not available, logging to console only")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        training_config = self.config["training"]
        grad_accumulation = training_config["grad_accumulation"]
        log_every = self.config["logging"]["log_every"]
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(), labels) / grad_accumulation
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), labels) / grad_accumulation
                loss.backward()
            
            total_loss += loss.item() * grad_accumulation
            
            # Collect predictions
            with torch.no_grad():
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                all_preds.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
                all_labels.extend(labels.cpu().numpy().tolist())
            
            # Gradient accumulation step
            if (batch_idx + 1) % grad_accumulation == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log to TensorBoard
                if self.writer and self.global_step % log_every == 0:
                    self.writer.add_scalar("train/loss", loss.item() * grad_accumulation, self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * grad_accumulation:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
        
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        
        # Handle edge cases
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = 0.5
        
        binary_preds = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "auc": auc,
            "f1": f1,
        }
    
    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Validate model.
        
        Args:
            loader: DataLoader to use (default: validation loader)
        
        Returns:
            Dictionary of validation metrics
        """
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_preds.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            all_labels.extend(labels.cpu().numpy().tolist())
        
        # Compute metrics
        avg_loss = total_loss / len(loader)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
        else:
            auc = 0.5
        
        binary_preds = (all_preds > 0.5).astype(int)
        accuracy = accuracy_score(all_labels, binary_preds)
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "auc": auc,
            "f1": f1,
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            metrics: Current metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
            "metrics": metrics,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only the N best checkpoints."""
        keep_best = self.config["checkpoints"]["keep_best"]
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        for ckpt in checkpoints[keep_best:]:
            ckpt.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def train(self):
        """Run full training loop."""
        training_config = self.config["training"]
        epochs = training_config["epochs"]
        patience = training_config["early_stop_patience"]
        min_delta = training_config["min_delta"]
        
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Log metrics
            print(f"\nEpoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            if self.writer:
                for name, value in train_metrics.items():
                    self.writer.add_scalar(f"train/{name}", value, epoch)
                for name, value in val_metrics.items():
                    self.writer.add_scalar(f"val/{name}", value, epoch)
            
            # Check for improvement
            current_metric = val_metrics["auc"]
            is_best = current_metric > self.best_metric + min_delta
            
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
                print(f"  New best AUC: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement for {self.patience_counter} epochs")
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        print("\n" + "="*80)
        print(f"Training complete in {total_time / 3600:.1f} hours")
        print(f"Best validation AUC: {self.best_metric:.4f}")
        print("="*80)
        
        # Final test evaluation
        print("\nEvaluating on test set...")
        test_metrics = self.validate(self.test_loader)
        print(f"Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, "
              f"AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        # Save final metrics
        final_metrics = {
            "best_val_auc": self.best_metric,
            "test_metrics": test_metrics,
            "total_epochs": self.current_epoch + 1,
            "training_time_hours": total_time / 3600,
        }
        
        with open(self.checkpoint_dir / "training_results.json", "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        # Update best_model.pt with test_metrics so API shows test accuracy
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            try:
                best_ckpt = torch.load(best_path, map_location=self.device)
                if isinstance(best_ckpt, dict):
                    best_ckpt["test_metrics"] = test_metrics
                    torch.save(best_ckpt, best_path)
                    print(f"Updated {best_path} with test metrics (acc: {test_metrics['accuracy']:.4f})")
            except Exception as e:
                print(f"Could not update best checkpoint with test_metrics: {e}")
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--backbone", type=str, help="Model backbone")
    parser.add_argument("--datasets-dir", type=str, help="Datasets directory")
    parser.add_argument("--checkpoint-dir", type=str, help="Checkpoint directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
            # Deep merge
            for key, value in user_config.items():
                if key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value
    
    # Override with command line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.datasets_dir:
        config["data"]["datasets_dir"] = args.datasets_dir
    if args.checkpoint_dir:
        config["checkpoints"]["save_dir"] = args.checkpoint_dir
    
    # Fix paths if not in container
    if not Path("/app/datasets").exists():
        config["data"]["datasets_dir"] = "./datasets"
        config["checkpoints"]["save_dir"] = "./checkpoints"
        config["logging"]["log_dir"] = "./logs"
    
    print("Training Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
