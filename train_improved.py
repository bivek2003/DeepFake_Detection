#!/usr/bin/env python3
"""
Improved Phase 2 Training - More data, better training strategy
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import logging
from pathlib import Path
from tqdm import tqdm
import random

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from deepfake_detector.models.efficientnet_detector import EfficientNetDeepfakeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedDeepfakeDataset(Dataset):
    """Improved dataset with more aggressive augmentation"""
    
    def __init__(self, video_paths, labels, transform=None, frames_per_video=10):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_data = []
        self._extract_faces()
    
    def _extract_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
        
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
        logger.info(f"Extracting faces from {len(self.video_paths)} videos (increased sampling)...")
        
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
                
                # Sample more frames for better coverage
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

def load_all_videos(data_root):
    """Load ALL available videos (no limit)"""
    data_root = Path(data_root)
    video_paths = []
    labels = []
    
    # Real videos
    for orig_dir in [data_root / "original_sequences" / "actors" / "c23" / "videos",
                     data_root / "original_sequences" / "youtube" / "c23" / "videos"]:
        if orig_dir.exists():
            videos = list(orig_dir.glob("*.mp4"))
            logger.info(f"Real videos from {orig_dir.name}: {len(videos)}")
            video_paths.extend([str(v) for v in videos])
            labels.extend([0] * len(videos))
    
    # Fake videos - all methods
    for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection", "FaceShifter"]:
        fake_dir = data_root / "manipulated_sequences" / method / "c23" / "videos"
        if fake_dir.exists():
            videos = list(fake_dir.glob("*.mp4"))
            logger.info(f"Fake videos from {method}: {len(videos)}")
            video_paths.extend([str(v) for v in videos])
            labels.extend([1] * len(videos))
    
    logger.info(f"Total: {labels.count(0)} real, {labels.count(1)} fake videos")
    return video_paths, labels

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load ALL videos
    video_paths, labels = load_all_videos(args.data_root)
    
    if len(video_paths) < 100:
        logger.error("Not enough videos found!")
        return
    
    # Split
    from sklearn.model_selection import train_test_split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        video_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    logger.info(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test videos")
    
    # Transforms with heavy augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets with more frames per video
    train_dataset = ImprovedDeepfakeDataset(train_paths, train_labels, train_transform, frames_per_video=10)
    val_dataset = ImprovedDeepfakeDataset(val_paths, val_labels, val_transform, frames_per_video=5)
    test_dataset = ImprovedDeepfakeDataset(test_paths, test_labels, val_transform, frames_per_video=5)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Samples: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Model with dropout
    model = EfficientNetDeepfakeDetector('efficientnet_b0').to(device)
    
    # Training setup with warmup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr*10, epochs=args.epochs, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    best_acc = 0.0
    patience = 10
    no_improve = 0
    
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
            }, 'models/faceforensics_improved.pth')
            logger.info(f"BEST: {val_acc:.2f}%")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            logger.info("Early stopping")
            break
    
    # Test
    checkpoint = torch.load('models/faceforensics_improved.pth')
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
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Best Val: {best_acc:.2f}%")
    logger.info(f"Test Acc: {test_acc:.2f}%")
    logger.info(f"Test F1: {test_f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    if test_acc >= 75:
        logger.info("SUCCESS: Production-ready model!")
    else:
        logger.info(f"Result: {test_acc:.2f}% - May need more data or training")

if __name__ == "__main__":
    main()
