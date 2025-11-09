#!/usr/bin/env python3
"""
FIXED VERSION: Addresses class imbalance and overfitting issues
Key changes marked with # FIX comments
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import logging
from pathlib import Path
import json
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
import timm
from collections import defaultdict
import re

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training_identity_split_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EfficientNetLSTM(nn.Module):
    def __init__(self, num_classes=1, lstm_hidden=256, lstm_layers=2, dropout=0.5):
        super(EfficientNetLSTM, self).__init__()
        # FIX 1: Use smaller, faster backbone to reduce overfitting
        self.backbone = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=0)
        
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
        
        # FIX 2: Simpler classifier head to reduce overfitting
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
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
    # FIX 3: Adjust focal loss parameters for better balance
    def __init__(self, alpha=0.5, gamma=2.0):  # Changed alpha from 0.25 to 0.5
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma = alpha, gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DeepFakeDataset(Dataset):
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
        
        # FIX 4: Better frame sampling strategy
        frame_indices = np.linspace(0, total_frames - 1, 
                                   min(self.frames_per_video * 2, total_frames), 
                                   dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                if self.face_detector:
                    boxes, _ = self.face_detector.detect(frame_rgb)
                    if boxes is not None and len(boxes) > 0:
                        box = boxes[0].astype(int)
                        x1, y1, x2, y2 = box
                        margin = 20
                        x1, y1 = max(0, x1-margin), max(0, y1-margin)
                        x2, y2 = min(frame_rgb.shape[1], x2+margin), min(frame_rgb.shape[0], y2+margin)
                        face = frame_rgb[y1:y2, x1:x2]
                        if face.shape[0] > 0 and face.shape[1] > 0:
                            frames.append(cv2.resize(face, (self.image_size, self.image_size)))
                            if len(frames) >= self.frames_per_video:
                                break
                            continue
            except:
                pass
            
            # Fallback: center crop
            h, w = frame_rgb.shape[:2]
            size = min(h, w)
            y1, x1 = (h-size)//2, (w-size)//2
            frames.append(cv2.resize(frame_rgb[y1:y1+size, x1:x1+size], 
                                    (self.image_size, self.image_size)))
            if len(frames) >= self.frames_per_video:
                break
        
        cap.release()
        return frames

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        faces = self.extract_faces_from_video(self.video_paths[idx])
        
        if len(faces) == 0:
            return torch.zeros(self.frames_per_video, 3, self.image_size, self.image_size), self.labels[idx]
        
        if len(faces) < self.frames_per_video:
            faces = faces + [faces[-1]] * (self.frames_per_video - len(faces))
        else:
            faces = faces[:self.frames_per_video]
        
        transformed = [self.transform(image=f)['image'] if self.transform else f for f in faces]
        return torch.stack(transformed), self.labels[idx]


def get_transforms(image_size=224):
    # FIX 5: More aggressive augmentation for training (fixed for albumentations v1.4+)
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=25, p=0.5),
        # Critical for deepfakes: compression artifacts
        A.OneOf([
            A.ImageCompression(quality_range=(10, 70), p=1.0),
            A.Downscale(scale_range=(0.2, 0.5), interpolation=cv2.INTER_LINEAR, p=1.0)
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(var_range=(20.0, 100.0), p=1.0),
            A.ISONoise(color_shift=(0.05, 0.3), intensity=(0.2, 0.7), p=1.0)
        ], p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 11), p=1.0),
            A.MotionBlur(blur_limit=11, p=1.0)
        ], p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.7),
        # Add random erasing
        A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(16, 32), hole_width_range=(16, 32), fill_value=0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


# FIX 6: Add gradient clipping and better monitoring
def train_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (frames, labels) in enumerate(pbar):
        frames = frames.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs, _ = model(frames)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # FIX: Add gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{running_loss/(batch_idx+1):.4f}'
            })
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1


# FIX 7: Find optimal threshold during validation
def validate(model, dataloader, criterion, device, threshold=0.5):
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
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs > threshold).float().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1, all_probs, cm


def find_optimal_threshold(labels, probs):
    """Find threshold that maximizes F1 score"""
    from sklearn.metrics import f1_score
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        preds = (np.array(probs) > threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"\n🎯 Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold


def prepare_identity_split_data(data_root):
    """Split by actor ID from filenames"""
    logger.info("="*70)
    logger.info("IDENTITY-BASED SPLIT")
    logger.info("="*70)

    data_root = Path(data_root)
    actor_videos = defaultdict(list)

    original_path = data_root / 'original_sequences'
    for subdir in ['actors', 'youtube']:
        video_dir = original_path / subdir / 'c23' / 'videos'
        if video_dir.exists():
            for video_file in video_dir.glob('*.mp4'):
                match = re.match(r'(\d+)__', video_file.stem)
                if match:
                    actor_id = match.group(1)
                    actor_videos[actor_id].append(str(video_file))

    logger.info(f"\n✅ Found {len(actor_videos)} unique actors in original videos")

    all_actors = list(actor_videos.keys())
    random.shuffle(all_actors)

    n_train = int(len(all_actors) * 0.70)
    n_val = int(len(all_actors) * 0.15)

    train_actors = set(all_actors[:n_train])
    val_actors = set(all_actors[n_train:n_train+n_val])
    test_actors = set(all_actors[n_train+n_val:])

    logger.info(f"\n✅ Actor split:")
    logger.info(f"   Train: {len(train_actors)} actors")
    logger.info(f"   Val:   {len(val_actors)} actors")
    logger.info(f"   Test:  {len(test_actors)} actors")

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for actor_id, videos in actor_videos.items():
        if actor_id in train_actors:
            X_train.extend(videos)
            y_train.extend([0] * len(videos))
        elif actor_id in val_actors:
            X_val.extend(videos)
            y_val.extend([0] * len(videos))
        else:
            X_test.extend(videos)
            y_test.extend([0] * len(videos))

    manipulated_path = data_root / 'manipulated_sequences'
    for method in ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 'FaceShifter', 'DeepFakeDetection']:
        method_dir = manipulated_path / method / 'c23' / 'videos'
        if method_dir.exists():
            for video_file in method_dir.glob('*.mp4'):
                match = re.match(r'(\d+)_(\d+)', video_file.stem)
                if match:
                    actor1, actor2 = match.group(1), match.group(2)

                    if actor1 in train_actors or actor2 in train_actors:
                        X_train.append(str(video_file))
                        y_train.append(1)
                    elif actor1 in val_actors or actor2 in val_actors:
                        X_val.append(str(video_file))
                        y_val.append(1)
                    elif actor1 in test_actors or actor2 in test_actors:
                        X_test.append(str(video_file))
                        y_test.append(1)

    # Shuffle
    for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]:
        combined = list(zip(X, y))
        random.shuffle(combined)
        if combined:
            X[:], y[:] = zip(*combined)

    logger.info(f"\n✅ Final dataset (NO ACTOR OVERLAP):")
    logger.info(f"   Train: {len(X_train)} ({y_train.count(0)} real, {y_train.count(1)} fake)")
    logger.info(f"   Val:   {len(X_val)} ({y_val.count(0)} real, {y_val.count(1)} fake)")
    logger.info(f"   Test:  {len(X_test)} ({y_test.count(0)} real, {y_test.count(1)} fake)")

    return list(X_train), list(X_val), list(X_test), list(y_train), list(y_val), list(y_test)


# FIX 8: Add weighted sampling for class balance
def get_weighted_sampler(labels):
    """Create weighted sampler to balance classes"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    
    logger.info(f"\n🎯 Class balancing:")
    logger.info(f"   Real:  {class_counts[0]} samples (weight: {class_weights[0]:.4f})")
    logger.info(f"   Fake:  {class_counts[1]} samples (weight: {class_weights[1]:.4f})")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_identity_split_data('datasets')

    face_detector = MTCNN(keep_all=False, device=device, post_process=False)
    train_transform, val_transform = get_transforms(224)

    train_dataset = DeepFakeDataset(X_train, y_train, train_transform, 10, face_detector, 224)
    val_dataset = DeepFakeDataset(X_val, y_val, val_transform, 10, face_detector, 224)
    test_dataset = DeepFakeDataset(X_test, y_test, val_transform, 10, face_detector, 224)

    # FIX 9: Use weighted sampler instead of regular shuffle
    train_sampler = get_weighted_sampler(y_train)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,  # Keep batch size small due to LSTM
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = EfficientNetLSTM(dropout=0.5).to(device)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    
    # FIX 10: Lower learning rate
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)  # Lower LR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()

    logger.info("\n" + "="*70)
    logger.info("🚀 STARTING TRAINING (IDENTITY-SPLIT WITH FIXES)")
    logger.info("="*70)
    
    best_val_f1 = 0.0  # FIX: Track F1 instead of accuracy
    patience_counter = 0
    best_threshold = 0.5

    for epoch in range(50):
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 Epoch {epoch+1}/50")
        logger.info(f"   Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info('='*70)
        
        # Train
        train_loss, train_acc, train_p, train_r, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch+1
        )
        
        logger.info(f"\n📈 Training Results:")
        logger.info(f"   Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"   Prec: {train_p:.4f} | Rec: {train_r:.4f} | F1: {train_f1:.4f}")
        
        # Validate with current threshold
        val_loss, val_acc, val_p, val_r, val_f1, val_probs, cm = validate(
            model, val_loader, criterion, device, threshold=0.5  # Always use 0.5 during training
        )
        
        logger.info(f"\n📊 Validation Results:")
        logger.info(f"   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        logger.info(f"   Prec: {val_p:.4f} | Rec: {val_r:.4f} | F1: {val_f1:.4f}")
        logger.info(f"\n   Confusion Matrix:")
        logger.info(f"   [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
        logger.info(f"    [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")
        
        # Calculate balanced accuracy (better metric for imbalanced data)
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate (recall)
        balanced_acc = (specificity + sensitivity) / 2
        
        logger.info(f"   Specificity (TNR): {specificity:.4f}")
        logger.info(f"   Sensitivity (TPR): {sensitivity:.4f}") 
        logger.info(f"   Balanced Accuracy: {balanced_acc:.4f}")
        
        # Update learning rate based on F1
        scheduler.step(val_f1)
        
        # FIX 11: Save best model based on BALANCED metrics
        # Use balanced accuracy instead of just F1 to account for class imbalance
        if balanced_acc > best_val_f1:
            best_val_f1 = balanced_acc
            patience_counter = 0
            
            # Find optimal threshold
            best_threshold = find_optimal_threshold(y_val, val_probs)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_accuracy': val_acc,
                'best_threshold': best_threshold
            }, 'models/best_model_identity_split.pth')
            
            logger.info(f"\n✅ NEW BEST MODEL SAVED!")
            logger.info(f"   F1: {best_val_f1:.4f}")
            logger.info(f"   Threshold: {best_threshold:.2f}")
        else:
            patience_counter += 1
            logger.info(f"\n⏳ No improvement for {patience_counter} epochs")
        
        if patience_counter >= 7:  # Reduced from 10
            logger.info("\n🛑 Early stopping triggered")
            break

    # Test with best model
    logger.info("\n" + "="*70)
    logger.info("🧪 FINAL TEST EVALUATION")
    logger.info("="*70)
    
    checkpoint = torch.load('models/best_model_identity_split.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_threshold = checkpoint.get('best_threshold', 0.5)
    
    logger.info(f"\n📌 Using threshold: {best_threshold:.2f}")
    
    test_loss, test_acc, test_p, test_r, test_f1, test_probs, test_cm = validate(
        model, test_loader, criterion, device, threshold=best_threshold
    )

    logger.info(f"\n✅ FINAL TEST RESULTS (unseen actors):")
    logger.info(f"   Accuracy:  {test_acc:.4f}")
    logger.info(f"   Precision: {test_p:.4f}")
    logger.info(f"   Recall:    {test_r:.4f}")
    logger.info(f"   F1-Score:  {test_f1:.4f}")
    logger.info(f"\n   Confusion Matrix:")
    logger.info(f"   [[TN={test_cm[0,0]:4d}  FP={test_cm[0,1]:4d}]")
    logger.info(f"    [FN={test_cm[1,0]:4d}  TP={test_cm[1,1]:4d}]]")
    logger.info(f"\n💾 Model saved to: models/best_model_identity_split.pth")


if __name__ == '__main__':
    main()
