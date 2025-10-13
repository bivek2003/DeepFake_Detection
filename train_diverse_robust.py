#!/usr/bin/env python3
"""
Robust training on ALL FaceForensics++ manipulation methods
Plus heavy augmentation to generalize to internet videos
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import logging
from pathlib import Path
from tqdm import tqdm
import random

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from deepfake_detector.models.efficientnet_detector import EfficientNetDeepfakeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiverseDeepfakeDataset(Dataset):
    """Dataset with heavy augmentation for real-world robustness"""
    
    def __init__(self, video_paths, labels, transform=None, frames_per_video=8):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_data = []
        self._extract_faces()
    
    def _extract_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            padding = int(0.2 * max(w, h))
            x1, y1 = max(0, x - padding), max(0, y - padding)
            x2, y2 = min(frame.shape[1], x + w + padding), min(frame.shape[0], y + h + padding)
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                face = cv2.resize(face, (224, 224))
                return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return None
    
    def _extract_faces(self):
        logger.info(f"Extracting faces from {len(self.video_paths)} videos...")
        
        for idx, video_path in enumerate(tqdm(self.video_paths)):
            label = self.labels[idx]
            
            if not os.path.exists(video_path):
                continue
            
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames == 0:
                    cap.release()
                    continue
                
                # Sample frames from different parts of video
                frame_indices = np.linspace(0, total_frames-1, min(self.frames_per_video * 3, total_frames), dtype=int)
                
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
                logger.warning(f"Error: {video_path}: {e}")
        
        logger.info(f"Total samples: {len(self.face_data)} (Real: {sum(1 for _, l in self.face_data if l == 0)}, Fake: {sum(1 for _, l in self.face_data if l == 1)})")
    
    def __len__(self):
        return len(self.face_data)
    
    def __getitem__(self, idx):
        face, label = self.face_data[idx]
        face = Image.fromarray(face)
        
        if self.transform:
            face = self.transform(face)
        
        return face, torch.tensor(label, dtype=torch.long)

def load_all_faceforensics_data(data_root):
    """Load ALL videos from ALL manipulation methods"""
    data_root = Path(data_root)
    video_paths = []
    labels = []
    
    logger.info("Loading ALL FaceForensics++ data...")
    
    # Real videos from both sources
    for source in ['actors', 'youtube']:
        real_dir = data_root / "original_sequences" / source / "c23" / "videos"
        if real_dir.exists():
            videos = list(real_dir.glob("*.mp4"))
            logger.info(f"Real videos from {source}: {len(videos)}")
            video_paths.extend([str(v) for v in videos])
            labels.extend([0] * len(videos))
    
    # Fake videos from ALL manipulation methods
    fake_methods = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    
    for method in fake_methods:
        fake_dir = data_root / "manipulated_sequences" / method / "c23" / "videos"
        if fake_dir.exists():
            videos = list(fake_dir.glob("*.mp4"))
            logger.info(f"Fake videos from {method}: {len(videos)}")
            video_paths.extend([str(v) for v in videos])
            labels.extend([1] * len(videos))
    
    logger.info(f"Total dataset: {labels.count(0)} real, {labels.count(1)} fake videos")
    return video_paths, labels

def create_robust_transforms():
    """Heavy augmentation to simulate real-world variations"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # More aggressive crop
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),  # More rotation
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Heavy color jitter
        transforms.RandomGrayscale(p=0.15),  # Some grayscale
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3),  # Random erasing for robustness
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load ALL videos
    video_paths, labels = load_all_faceforensics_data(args.data_root)
    
    if len(video_paths) < 100:
        logger.error("Not enough videos!")
        return
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels, test_size=0.25, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    logger.info(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test videos")
    
    # Create datasets with heavy augmentation
    train_transform, val_transform = create_robust_transforms()
    
    logger.info("Creating training dataset (this will take time)...")
    train_dataset = DiverseDeepfakeDataset(train_paths, train_labels, train_transform, frames_per_video=8)
    
    logger.info("Creating validation dataset...")
    val_dataset = DiverseDeepfakeDataset(val_paths, val_labels, val_transform, frames_per_video=4)
    
    logger.info("Creating test dataset...")
    test_dataset = DiverseDeepfakeDataset(test_paths, test_labels, val_transform, frames_per_video=4)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Dataset ready: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")
    
    # Model with more dropout for better generalization
    model = EfficientNetDeepfakeDetector('efficientnet_b0').to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)  # More label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.08)  # Higher weight decay
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr*15, epochs=args.epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.15, div_factor=25
    )
    
    best_acc = 0.0
    patience = 15
    no_improve = 0
    
    logger.info("Starting robust training for real-world generalization...")
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            train_total += target.size(0)
            train_correct += (pred == target).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.3f}', 'Acc': f'{100*train_correct/train_total:.1f}%'})
        
        train_acc = 100.0 * train_correct / train_total
        
        # Validate
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, pred = torch.max(output, 1)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds, average='binary')
        
        logger.info(f"Epoch {epoch+1}: Train {train_acc:.2f}%, Val {val_acc:.2f}%, F1 {val_f1:.4f}")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'f1': val_f1,
                'epoch': epoch
            }, 'models/faceforensics_robust.pth')
            logger.info(f"BEST: {val_acc:.2f}%")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            logger.info("Early stopping")
            break
    
    # Test
    checkpoint = torch.load('models/faceforensics_robust.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_targets = [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    test_acc = accuracy_score(test_targets, test_preds) * 100
    test_f1 = f1_score(test_targets, test_preds, average='binary')
    cm = confusion_matrix(test_targets, test_preds)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ROBUST MODEL - FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Best Val: {best_acc:.2f}%")
    logger.info(f"Test Acc: {test_acc:.2f}%")
    logger.info(f"Test F1: {test_f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Model saved: models/faceforensics_robust.pth")
    
    if test_acc >= 85:
        logger.info("SUCCESS: Robust model ready for real-world deployment!")

if __name__ == "__main__":
    main()
