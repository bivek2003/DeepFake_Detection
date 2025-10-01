#!/usr/bin/env python3
"""
Phase 2: Real Deepfake Detection Training - FaceForensics++
Updated for your directory structure
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from deepfake_detector.models.efficientnet_detector import EfficientNetDeepfakeDetector
from deepfake_detector.models.xception_detector import XceptionDeepfakeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDeepfakeDataset(Dataset):
    """Dataset for real deepfake videos with face extraction"""
    
    def __init__(self, video_paths, labels, transform=None, frames_per_video=5):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Extract faces during initialization
        self.face_data = []
        self._extract_faces()
    
    def _extract_face(self, frame):
        """Extract largest face from frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        
        if len(faces) > 0:
            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Add padding
            padding = int(0.3 * max(w, h))
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
            
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                face = cv2.resize(face, (224, 224))
                return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Fallback: resize entire frame
        return cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
    
    def _extract_faces(self):
        """Extract faces from all videos"""
        logger.info(f"Extracting faces from {len(self.video_paths)} videos...")
        
        for idx, video_path in enumerate(tqdm(self.video_paths)):
            label = self.labels[idx]
            
            if not os.path.exists(video_path):
                logger.warning(f"Video not found: {video_path}")
                continue
            
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames == 0:
                    cap.release()
                    continue
                
                # Sample frames evenly
                frame_indices = np.linspace(
                    int(total_frames * 0.1),  # Skip first 10%
                    int(total_frames * 0.9),  # Skip last 10%
                    min(self.frames_per_video * 2, total_frames),
                    dtype=int
                )
                
                faces_extracted = 0
                for frame_idx in frame_indices:
                    if faces_extracted >= self.frames_per_video:
                        break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        face = self._extract_face(frame)
                        if face is not None:
                            self.face_data.append((face, label))
                            faces_extracted += 1
                
                cap.release()
                
            except Exception as e:
                logger.warning(f"Error processing {video_path}: {e}")
                continue
        
        logger.info(f"Extracted {len(self.face_data)} face samples total")
        logger.info(f"Real: {sum(1 for _, l in self.face_data if l == 0)}, "
                   f"Fake: {sum(1 for _, l in self.face_data if l == 1)}")
    
    def __len__(self):
        return len(self.face_data)
    
    def __getitem__(self, idx):
        face, label = self.face_data[idx]
        face = Image.fromarray(face)
        
        if self.transform:
            face = self.transform(face)
        
        return face, torch.tensor(label, dtype=torch.long)

def load_faceforensics_data(data_root, max_videos_per_category=200):
    """Load FaceForensics++ with your directory structure"""
    data_root = Path(data_root)
    video_paths = []
    labels = []
    
    logger.info("Loading FaceForensics++ dataset...")
    
    # Original videos (real) - using actors
    original_dirs = [
        data_root / "original_sequences" / "actors" / "c23" / "videos",
        data_root / "original_sequences" / "youtube" / "c23" / "videos"
    ]
    
    original_videos = []
    for orig_dir in original_dirs:
        if orig_dir.exists():
            videos = list(orig_dir.glob("*.mp4"))
            original_videos.extend(videos)
            logger.info(f"Found {len(videos)} videos in {orig_dir}")
    
    # Sample if too many
    if len(original_videos) > max_videos_per_category:
        original_videos = random.sample(original_videos, max_videos_per_category)
    
    for video_file in original_videos:
        video_paths.append(str(video_file))
        labels.append(0)  # Real
    
    logger.info(f"Loaded {len(original_videos)} real videos")
    
    # Manipulated videos (fake)
    fake_methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection", "FaceShifter"]
    
    for method in fake_methods:
        fake_dir = data_root / "manipulated_sequences" / method / "c23" / "videos"
        if fake_dir.exists():
            fake_videos = list(fake_dir.glob("*.mp4"))
            
            # Sample if too many
            if len(fake_videos) > max_videos_per_category // len(fake_methods):
                fake_videos = random.sample(
                    fake_videos, 
                    max_videos_per_category // len(fake_methods)
                )
            
            logger.info(f"Found {len(fake_videos)} videos in {method}")
            
            for video_file in fake_videos:
                video_paths.append(str(video_file))
                labels.append(1)  # Fake
    
    logger.info(f"Total: {labels.count(0)} real, {labels.count(1)} fake videos")
    return video_paths, labels

def create_transforms():
    """Data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100*correct/total:.1f}%'})
    
    return running_loss / len(loader), 100.0 * correct / total

def validate(model, loader, device):
    """Validate model"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            predictions.extend(predicted.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(targets, predictions) * 100
    f1 = f1_score(targets, predictions, average='binary')
    
    return accuracy, f1, predictions, targets

def plot_results(train_losses, val_accs, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(val_accs, 'g-', label='Val Accuracy')
    ax2.axhline(y=87, color='r', linestyle='--', label='Target (87%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets', help='Dataset root')
    parser.add_argument('--max_videos', type=int, default=200, help='Max videos per category')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='efficientnet', choices=['efficientnet', 'xception', 'both'])
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    video_paths, labels = load_faceforensics_data(args.data_root, args.max_videos)
    
    if len(video_paths) == 0:
        logger.error("No videos found! Check your dataset path.")
        return
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        video_paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    
    logger.info(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    
    # Create datasets
    train_transform, val_transform = create_transforms()
    
    logger.info("Creating training dataset...")
    train_dataset = RealDeepfakeDataset(train_paths, train_labels, train_transform, frames_per_video=5)
    
    logger.info("Creating validation dataset...")
    val_dataset = RealDeepfakeDataset(val_paths, val_labels, val_transform, frames_per_video=3)
    
    logger.info("Creating test dataset...")
    test_dataset = RealDeepfakeDataset(test_paths, test_labels, val_transform, frames_per_video=3)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    logger.info(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    
    # Train model
    model = EfficientNetDeepfakeDetector('efficientnet_b0').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    best_acc = 0.0
    train_losses = []
    val_accs = []
    
    logger.info(f"Training EfficientNet-B0 for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_acc, val_f1, _, _ = validate(model, val_loader, device)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'f1_score': val_f1,
            }, 'models/faceforensics_best.pth')
            
            status = "TARGET REACHED!" if val_acc >= 87 else "NEW BEST"
            logger.info(f"{status} Saved model with {val_acc:.2f}% accuracy")
    
    # Test evaluation
    logger.info("\nEvaluating on test set...")
    test_acc, test_f1, predictions, targets = validate(model, test_loader, device)
    
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Plot
    plot_results(train_losses, val_accs, 'results/training_curves.png')
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Test Accuracy: {test_acc:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    # Final results
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 2 COMPLETE - REAL DEEPFAKE DETECTION")
    logger.info(f"{'='*60}")
    logger.info(f"Best Validation Accuracy: {best_acc:.2f}%")
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info(f"Model saved: models/faceforensics_best.pth")
    
    if test_acc >= 87:
        logger.info("SUCCESS: Model ready for Phase 3 deployment!")
    else:
        logger.info(f"Performance: {test_acc:.2f}% (Realistic for production)")

if __name__ == "__main__":
    main()
