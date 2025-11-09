#!/usr/bin/env python3
"""
Complete Standalone Balanced Training Script
No external dependencies - everything included
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
import timm

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_balanced.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class EfficientNetLSTM(nn.Module):
    """EfficientNet + BiLSTM + Attention"""
    
    def __init__(self, num_classes=1, lstm_hidden=256, lstm_layers=2, dropout=0.5):
        super(EfficientNetLSTM, self).__init__()
        
        self.backbone = timm.create_model('tf_efficientnetv2_m', pretrained=True, num_classes=0)
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, num_frames, -1)
        
        lstm_out, _ = self.lstm(features)
        
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        context = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.classifier(context)
        
        return output, attention_weights


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ============================================================================
# DATASET
# ============================================================================

class DeepFakeDataset(Dataset):
    """Dataset with face extraction"""
    
    def __init__(self, video_paths, labels, transform=None, frames_per_video=10, 
                 face_detector=None, image_size=224):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.face_detector = face_detector
        self.image_size = image_size
        
    def extract_faces_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        frame_indices = np.linspace(0, total_frames - 1, 
                                   min(self.frames_per_video * 3, total_frames), 
                                   dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.face_detector is not None:
                try:
                    boxes, _ = self.face_detector.detect(frame_rgb)
                    if boxes is not None and len(boxes) > 0:
                        box = boxes[0].astype(int)
                        x1, y1, x2, y2 = box
                        margin = 20
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(frame_rgb.shape[1], x2 + margin)
                        y2 = min(frame_rgb.shape[0], y2 + margin)
                        
                        face = frame_rgb[y1:y2, x1:x2]
                        if face.shape[0] > 0 and face.shape[1] > 0:
                            face = cv2.resize(face, (self.image_size, self.image_size))
                            frames.append(face)
                except:
                    h, w = frame_rgb.shape[:2]
                    size = min(h, w)
                    y1 = (h - size) // 2
                    x1 = (w - size) // 2
                    face = frame_rgb[y1:y1+size, x1:x1+size]
                    face = cv2.resize(face, (self.image_size, self.image_size))
                    frames.append(face)
            else:
                h, w = frame_rgb.shape[:2]
                size = min(h, w)
                y1 = (h - size) // 2
                x1 = (w - size) // 2
                face = frame_rgb[y1:y1+size, x1:x1+size]
                face = cv2.resize(face, (self.image_size, self.image_size))
                frames.append(face)
            
            if len(frames) >= self.frames_per_video:
                break
        
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        faces = self.extract_faces_from_video(video_path)
        
        if len(faces) == 0:
            return torch.zeros(self.frames_per_video, 3, self.image_size, self.image_size), label
        
        if len(faces) < self.frames_per_video:
            faces = faces + [faces[-1]] * (self.frames_per_video - len(faces))
        else:
            faces = faces[:self.frames_per_video]
        
        transformed_faces = []
        for face in faces:
            if self.transform:
                transformed = self.transform(image=face)
                face = transformed['image']
            transformed_faces.append(face)
        
        return torch.stack(transformed_faces), label


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(image_size=224):
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.OneOf([
            A.ImageCompression(quality_lower=15, quality_upper=70, p=1.0),
            A.Downscale(scale_min=0.25, scale_max=0.6, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(20.0, 80.0)),
            A.ISONoise(color_shift=(0.05, 0.25), intensity=(0.2, 0.6)),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9)),
            A.MotionBlur(blur_limit=9),
        ], p=0.4),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.6),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for frames, labels in pbar:
        frames = frames.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs, _ = model(frames)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for frames, labels in pbar:
            frames = frames.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            
            with torch.cuda.amp.autocast():
                outputs, _ = model(frames)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return epoch_loss, accuracy, precision, recall, f1, all_probs


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_balanced_data(data_root):
    logger.info("="*70)
    logger.info("PREPARING BALANCED DATASET")
    logger.info("="*70)
    
    data_root = Path(data_root)
    
    # Collect real videos
    real_videos = []
    original_path = data_root / 'original_sequences'
    for subdir in ['actors', 'youtube']:
        video_dir = original_path / subdir / 'c23' / 'videos'
        if video_dir.exists():
            videos = [str(f) for f in video_dir.glob('*.mp4')]
            logger.info(f"  {subdir}: {len(videos)} videos")
            real_videos.extend(videos)
    
    internet_real_path = data_root / 'internet_real_videos'
    if internet_real_path.exists():
        videos = [str(f) for f in internet_real_path.glob('*.mp4')]
        logger.info(f"  internet: {len(videos)} videos")
        real_videos.extend(videos)
    
    random.shuffle(real_videos)
    
    # Collect fake videos
    fake_videos = []
    manipulated_path = data_root / 'manipulated_sequences'
    for method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 
                   'FaceShifter', 'DeepFakeDetection']:
        method_dir = manipulated_path / method / 'c23' / 'videos'
        if method_dir.exists():
            videos = [str(f) for f in method_dir.glob('*.mp4')]
            logger.info(f"  {method}: {len(videos)} videos")
            fake_videos.extend(videos)
    
    random.shuffle(fake_videos)
    
    logger.info(f"\nTotal: Real={len(real_videos)}, Fake={len(fake_videos)}")
    
    # Balance
    num_per_class = len(real_videos)
    fake_videos_sampled = random.sample(fake_videos, num_per_class)
    
    logger.info(f"\n✅ BALANCED: {num_per_class} real + {num_per_class} fake = {num_per_class*2} total")
    
    # Combine and shuffle
    video_paths = real_videos + fake_videos_sampled
    labels = [0] * len(real_videos) + [1] * len(fake_videos_sampled)
    
    combined = list(zip(video_paths, labels))
    random.shuffle(combined)
    video_paths, labels = zip(*combined)
    video_paths, labels = list(video_paths), list(labels)
    
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        video_paths, labels, test_size=0.15, random_state=SEED, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp
    )
    
    logger.info(f"\nSplits:")
    logger.info(f"  Train: {len(X_train)} ({y_train.count(0)} real, {y_train.count(1)} fake)")
    logger.info(f"  Val:   {len(X_val)} ({y_val.count(0)} real, {y_val.count(1)} fake)")
    logger.info(f"  Test:  {len(X_test)} ({y_test.count(0)} real, {y_test.count(1)} fake)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_balanced_data('datasets')
    
    # Setup
    face_detector = MTCNN(keep_all=False, device=device, post_process=False)
    train_transform, val_transform = get_transforms(224)
    
    # Datasets
    train_dataset = DeepFakeDataset(X_train, y_train, train_transform, 10, face_detector, 224)
    val_dataset = DeepFakeDataset(X_val, y_val, val_transform, 10, face_detector, 224)
    test_dataset = DeepFakeDataset(X_test, y_test, val_transform, 10, face_detector, 224)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = EfficientNetLSTM().to(device)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler()
    
    # Train
    logger.info("\nSTARTING TRAINING")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(50):
        logger.info(f"\nEpoch {epoch+1}/50")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_p, val_r, val_f1, _ = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        logger.info(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        logger.info(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'balanced': True
            }, 'models/best_model_balanced.pth')
            logger.info(f"✅ Saved best: {val_acc:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            logger.info("Early stopping")
            break
    
    # Test
    checkpoint = torch.load('models/best_model_balanced.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_p, test_r, test_f1, _ = validate(model, test_loader, criterion, device)
    
    logger.info(f"\n✅ TEST RESULTS:")
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  Precision: {test_p:.4f}")
    logger.info(f"  Recall:    {test_r:.4f}")
    logger.info(f"  F1-Score:  {test_f1:.4f}")
    
    logger.info("\n✅ DONE! Model saved to models/best_model_balanced.pth")


if __name__ == '__main__':
    main()
