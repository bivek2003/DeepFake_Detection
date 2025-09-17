#!/usr/bin/env python3
"""
Comprehensive training pipeline for deepfake detection models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)

class DeepfakeTrainer:
    """Comprehensive training pipeline"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_accuracy = 0.0
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        return running_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy()[:, 1])
        
        accuracy = 100.0 * correct / total
        val_loss /= len(val_loader)
        
        # Calculate additional metrics
        f1 = f1_score(all_targets, all_predictions, average='binary')
        auc = roc_auc_score(all_targets, all_probabilities)
        
        return val_loss, accuracy, f1, auc, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, epochs=20, save_path='best_model.pth'):
        """Full training pipeline"""
        logger.info("Starting training...")
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, accuracy, f1, auc, predictions, targets = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(accuracy)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {accuracy:.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # Save best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'auc': auc
                }, save_path)
                logger.info(f"New best model saved with accuracy: {accuracy:.2f}%")
        
        logger.info(f"Training completed. Best accuracy: {self.best_accuracy:.2f}%")
        return predictions, targets
