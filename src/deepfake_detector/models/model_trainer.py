"""
Comprehensive training pipeline integrating with existing data pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

class DeepfakeTrainer:
    """Training pipeline optimized for >87% accuracy target"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=learning_rate * 10,
            epochs=20,  # Will be updated in train()
            steps_per_epoch=100,  # Will be updated in train()
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        
    def train_epoch(self, train_loader):
        """Train for one epoch with mixed precision"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler()
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with autocasting
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100*correct/total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate model with comprehensive metrics"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                val_loss += loss.item()
                
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy()[:, 1])
        
        # Calculate metrics
        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions) * 100
        f1 = f1_score(all_targets, all_predictions, average='binary')
        auc = roc_auc_score(all_targets, all_probabilities)
        
        return val_loss, accuracy, f1, auc, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=20, save_dir='./models', model_name='deepfake_detector'):
        """Full training pipeline with early stopping"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Update scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'] * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Target: >87% accuracy")
        
        patience = 7
        no_improvement_count = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            logger.info("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_acc, val_f1, val_auc, predictions, targets = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            logger.info(f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
            
            # Save best model (prioritize accuracy, but consider F1)
            is_best = val_acc > self.best_accuracy
            
            if is_best:
                self.best_accuracy = val_acc
                self.best_f1 = val_f1
                no_improvement_count = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'accuracy': val_acc,
                    'f1_score': val_f1,
                    'auc': val_auc,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies,
                }
                
                save_path = save_dir / f'best_{model_name}.pth'
                torch.save(checkpoint, save_path)
                
                status = "🎯 TARGET REACHED!" if val_acc >= 87 else "📈 NEW BEST"
                logger.info(f"{status} Model saved: Accuracy = {val_acc:.2f}%")
                
            else:
                no_improvement_count += 1
                
            # Early stopping
            if no_improvement_count >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            # Achievement check
            if val_acc >= 87.0:
                logger.info("🎉 TARGET ACCURACY ACHIEVED! (≥87%)")
        
        # Final summary
        logger.info(f"\nTraining completed!")
        logger.info(f"Best validation accuracy: {self.best_accuracy:.2f}%")
        logger.info(f"Best F1 score: {self.best_f1:.4f}")
        
        target_status = "✅ SUCCESS" if self.best_accuracy >= 87 else "❌ BELOW TARGET"
        logger.info(f"Target status: {target_status}")
        
        # Plot training curves
        self._plot_training_curves(save_dir)
        
        return predictions, targets
    
    def _plot_training_curves(self, save_dir):
        """Plot and save training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'bo-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'ro-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, self.val_accuracies, 'go-', label='Val Accuracy')
        ax2.axhline(y=87, color='r', linestyle='--', alpha=0.7, label='Target (87%)')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 Score curve
        ax3.plot(epochs, self.val_f1_scores, 'mo-', label='Val F1 Score')
        ax3.set_title('Validation F1 Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary text
        ax4.axis('off')
        summary_text = f"""
        TRAINING SUMMARY
        
        Best Accuracy: {self.best_accuracy:.2f}%
        Best F1 Score: {self.best_f1:.4f}
        Total Epochs: {len(epochs)}
        
        Target: ≥87% accuracy
        Status: {'✅ ACHIEVED' if self.best_accuracy >= 87 else '❌ NOT REACHED'}
        
        Model: {self.model.__class__.__name__}
        Device: {self.device}
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {save_dir}/training_curves.png")
