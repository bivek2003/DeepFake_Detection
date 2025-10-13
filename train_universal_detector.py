#!/usr/bin/env python3
"""
Universal Deepfake Detector Training
Works for traditional deepfakes AND modern AI-generated videos
Uses multi-scale analysis, frequency domain, and temporal consistency
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import logging
from pathlib import Path
from tqdm import tqdm
import random
import re
from collections import defaultdict
from scipy import fftpack
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent / 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalDeepfakeDetector(nn.Module):
    """
    Multi-branch detector that analyzes:
    1. Spatial features (RGB)
    2. Frequency domain (FFT)
    3. Temporal consistency
    """
    
    def __init__(self, backbone='efficientnet_b4'):
        super().__init__()
        
        # Import here to avoid circular dependency
        from deepfake_detector.models.efficientnet_detector import EfficientNetDeepfakeDetector
        
        # RGB Branch (spatial features)
        self.rgb_branch = EfficientNetDeepfakeDetector(backbone)
        
        # Frequency Branch (FFT features)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(2 + 128, 256),  # 2 from RGB, 128 from freq
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        
    def extract_frequency_features(self, x):
        """Extract frequency domain features using FFT"""
        # Convert to frequency domain
        batch_size = x.size(0)
        freq_features = []
        
        for i in range(batch_size):
            img = x[i].cpu().numpy().transpose(1, 2, 0)
            
            # Apply FFT to each channel
            fft_channels = []
            for c in range(3):
                fft = np.fft.fft2(img[:, :, c])
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)
                
                # Log scale and normalize
                magnitude = np.log(magnitude + 1)
                magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-8)
                fft_channels.append(magnitude)
            
            freq_img = np.stack(fft_channels, axis=0)
            freq_features.append(freq_img)
        
        freq_features = torch.FloatTensor(np.stack(freq_features)).to(x.device)
        return freq_features
    
    def forward(self, x):
        # RGB branch
        rgb_out = self.rgb_branch(x)
        
        # Frequency branch
        freq_features = self.extract_frequency_features(x)
        freq_conv_out = self.freq_conv(freq_features)
        freq_conv_out = freq_conv_out.view(freq_conv_out.size(0), -1)
        
        # Concatenate features
        combined = torch.cat([rgb_out, freq_conv_out], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        return output


class UniversalDeepfakeDataset(Dataset):
    """
    Enhanced dataset with:
    - Multi-scale face extraction
    - Quality filtering
    - Temporal frame sampling
    """
    
    def __init__(self, video_paths, labels, transform=None, frames_per_video=15):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        
        # Use DNN face detector (more robust than Haar Cascade)
        self.face_detector = self._load_face_detector()
        self.face_data = []
        self._extract_faces()
    
    def _load_face_detector(self):
        """Load DNN-based face detector"""
        model_path = "deploy.prototxt"
        weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        # Check if DNN model exists, fallback to Haar Cascade
        if os.path.exists(model_path) and os.path.exists(weights_path):
            logger.info("Using DNN face detector")
            return cv2.dnn.readNetFromCaffe(model_path, weights_path)
        else:
            logger.info("Using Haar Cascade face detector (download DNN model for better results)")
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def _detect_face_dnn(self, frame):
        """Detect face using DNN"""
        if isinstance(self.face_detector, cv2.dnn_Net):
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            best_confidence = 0
            best_box = None
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5 and confidence > best_confidence:
                    best_confidence = confidence
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_box = box.astype(int)
            
            return best_box
        else:
            # Fallback to Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
            if len(faces) > 0:
                return max(faces, key=lambda f: f[2] * f[3])
            return None
    
    def _extract_face(self, frame):
        """Extract face with quality check"""
        face_box = self._detect_face_dnn(frame)
        
        if face_box is not None:
            if len(face_box) == 4:
                if isinstance(self.face_detector, cv2.dnn_Net):
                    x1, y1, x2, y2 = face_box
                    w, h = x2 - x1, y2 - y1
                else:
                    x1, y1, w, h = face_box
                    x2, y2 = x1 + w, y1 + h
                
                # Add padding
                padding = int(0.3 * max(w, h))
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                face = frame[y1:y2, x1:x2]
                
                # Quality check - avoid blurry faces
                if face.size > 0:
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                    
                    # Only keep sharp faces (sharpness > 50)
                    if laplacian_var > 50:
                        face = cv2.resize(face, (224, 224))
                        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB), laplacian_var
        
        return None, 0
    
    def _extract_faces(self):
        logger.info(f"Extracting high-quality faces from {len(self.video_paths)} videos...")
        
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
                
                # Smart frame sampling - more frames from middle section
                start_frame = int(total_frames * 0.1)  # Skip first 10%
                end_frame = int(total_frames * 0.9)    # Skip last 10%
                sample_range = end_frame - start_frame
                
                frame_indices = np.linspace(start_frame, end_frame, 
                                          min(self.frames_per_video * 2, sample_range), 
                                          dtype=int)
                
                faces_with_quality = []
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        face, quality = self._extract_face(frame)
                        if face is not None:
                            faces_with_quality.append((face, quality))
                
                cap.release()
                
                # Keep only the best quality faces
                if faces_with_quality:
                    faces_with_quality.sort(key=lambda x: x[1], reverse=True)
                    best_faces = faces_with_quality[:self.frames_per_video]
                    
                    for face, _ in best_faces:
                        self.face_data.append((face, label))
                
            except Exception as e:
                logger.warning(f"Error processing {video_path}: {e}")
        
        logger.info(f"Extracted {len(self.face_data)} high-quality samples")
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


def load_all_videos_with_pairing(data_root):
    """Load videos with group awareness"""
    data_root = Path(data_root)
    video_groups = defaultdict(list)
    all_videos = []
    
    # Real videos - Actors
    actors_dir = data_root / "original_sequences" / "actors" / "c23" / "videos"
    if actors_dir.exists():
        videos = list(actors_dir.glob("*.mp4"))
        logger.info(f"Real videos from actors: {len(videos)}")
        for video in videos:
            match = re.match(r'(\d+)__', video.stem)
            if match:
                actor_id = match.group(1)
                group = f"actor_{actor_id}"
                all_videos.append({'path': str(video), 'label': 0, 'group': group})
    
    # Real videos - YouTube
    youtube_dir = data_root / "original_sequences" / "youtube" / "c23" / "videos"
    if youtube_dir.exists():
        videos = list(youtube_dir.glob("*.mp4"))
        logger.info(f"Real videos from youtube: {len(videos)}")
        for video in videos:
            video_id = video.stem
            group = f"youtube_{video_id}"
            all_videos.append({'path': str(video), 'label': 0, 'group': group})
    
    # Fake videos - All methods
    methods = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceSwap", 
               "NeuralTextures", "FaceShifter"]
    
    for method in methods:
        method_dir = data_root / "manipulated_sequences" / method / "c23" / "videos"
        if method_dir.exists():
            videos = list(method_dir.glob("*.mp4"))
            logger.info(f"Fake videos from {method}: {len(videos)}")
            
            for video in videos:
                if method == "DeepFakeDetection":
                    match = re.match(r'(\d+)_(\d+)__', video.stem)
                    if match:
                        group = f"actor_{match.group(1)}_{match.group(2)}"
                else:
                    match = re.match(r'(\d+)_(\d+)', video.stem)
                    if match:
                        group = f"youtube_{match.group(1)}_{match.group(2)}"
                
                if match:
                    all_videos.append({'path': str(video), 'label': 1, 'group': group})
    
    real = sum(1 for v in all_videos if v['label'] == 0)
    fake = sum(1 for v in all_videos if v['label'] == 1)
    logger.info(f"Total: {real} real, {fake} fake videos")
    
    return all_videos

def balance_and_split(all_videos, test_size=0.2, val_size=0.15):
    """Balance dataset and split by groups"""
    real_videos = [v for v in all_videos if v['label'] == 0]
    fake_videos = [v for v in all_videos if v['label'] == 1]
    
    min_samples = min(len(real_videos), len(fake_videos))
    random.seed(42)
    fake_videos = random.sample(fake_videos, min_samples)
    
    balanced = real_videos + fake_videos
    
    # Group-aware splitting
    groups = defaultdict(list)
    for v in balanced:
        groups[v['group']].append(v)
    
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    
    n = len(group_keys)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    
    test_groups = group_keys[:n_test]
    val_groups = group_keys[n_test:n_test + n_val]
    train_groups = group_keys[n_test + n_val:]
    
    train = [v for g in train_groups for v in groups[g]]
    val = [v for g in val_groups for v in groups[g]]
    test = [v for g in test_groups for v in groups[g]]
    
    logger.info(f"Balanced split: {len(train)} train, {len(val)} val, {len(test)} test")
    
    return train, val, test

def create_advanced_transforms():
    """Advanced augmentation for robustness"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--backbone', type=str, default='efficientnet_b4',
                       help='efficientnet_b0 to b7')
    args = parser.parse_args()
    
    os.makedirs('models', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Load and prepare data
    all_videos = load_all_videos_with_pairing(args.data_root)
    train_videos, val_videos, test_videos = balance_and_split(all_videos)
    
    train_paths = [v['path'] for v in train_videos]
    train_labels = [v['label'] for v in train_videos]
    val_paths = [v['path'] for v in val_videos]
    val_labels = [v['label'] for v in val_videos]
    test_paths = [v['path'] for v in test_videos]
    test_labels = [v['label'] for v in test_videos]
    
    # Create datasets
    train_transform, val_transform = create_advanced_transforms()
    
    train_dataset = UniversalDeepfakeDataset(train_paths, train_labels, 
                                            train_transform, frames_per_video=15)
    val_dataset = UniversalDeepfakeDataset(val_paths, val_labels,
                                          val_transform, frames_per_video=10)
    test_dataset = UniversalDeepfakeDataset(test_paths, test_labels,
                                           val_transform, frames_per_video=10)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Samples: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Model
    model = UniversalDeepfakeDetector(args.backbone).to(device)
    
    # Loss and optimizer
    real_count = train_labels.count(0)
    fake_count = train_labels.count(1)
    total = real_count + fake_count
    weight_real = total / (2 * real_count)
    weight_fake = total / (2 * fake_count)
    class_weights = torch.tensor([weight_real, weight_fake]).to(device)
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    best_f1 = 0.0
    patience = 15
    no_improve = 0
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING UNIVERSAL DEEPFAKE DETECTOR")
    logger.info("="*70 + "\n")
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = torch.max(output, 1)
            train_preds.extend(pred.cpu().numpy())
            train_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.3f}'})
        
        scheduler.step()
        
        train_acc = accuracy_score(train_targets, train_preds) * 100
        train_f1 = f1_score(train_targets, train_preds)
        
        # Validate
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                _, pred = torch.max(output, 1)
                
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())
                val_probs.extend(probs[:, 1].cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_preds) * 100
        val_f1 = f1_score(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_probs)
        val_cm = confusion_matrix(val_targets, val_preds)
        
        tn, fp, fn, tp = val_cm.ravel()
        real_acc = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
        fake_acc = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
        
        logger.info(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, F1 {train_f1:.4f}")
        logger.info(f"  Val: Acc {val_acc:.2f}%, F1 {val_f1:.4f}, AUC {val_auc:.4f}")
        logger.info(f"  Real {real_acc:.2f}%, Fake {fake_acc:.2f}%")
        
        # Save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'f1': val_f1,
                'auc': val_auc,
                'epoch': epoch
            }, 'models/universal_deepfake_detector.pth')
            logger.info(f"  ✓ BEST: F1 {val_f1:.4f}, AUC {val_auc:.4f}")
        else:
            no_improve += 1
        
        if no_improve >= patience:
            logger.info("Early stopping")
            break
    
    # Test
    logger.info("\n" + "="*70)
    logger.info("TESTING BEST MODEL")
    logger.info("="*70 + "\n")
    
    checkpoint = torch.load('models/universal_deepfake_detector.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, pred = torch.max(output, 1)
            
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
            test_probs.extend(probs[:, 1].cpu().numpy())
    
    test_acc = accuracy_score(test_targets, test_preds) * 100
    test_f1 = f1_score(test_targets, test_preds)
    test_auc = roc_auc_score(test_targets, test_probs)
    test_cm = confusion_matrix(test_targets, test_preds)
    
    tn, fp, fn, tp = test_cm.ravel()
    real_acc = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0
    fake_acc = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    logger.info(f"\n{'='*70}")
    logger.info("FINAL RESULTS - UNIVERSAL DEEPFAKE DETECTOR")
    logger.info(f"{'='*70}")
    logger.info(f"\nTest Metrics:")
    logger.info(f"  Accuracy: {test_acc:.2f}%")
    logger.info(f"  F1 Score: {test_f1:.4f}")
    logger.info(f"  AUC-ROC: {test_auc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"\nPer-Class:")
    logger.info(f"  Real: {real_acc:.2f}%")
    logger.info(f"  Fake: {fake_acc:.2f}%")
    logger.info(f"  Balance: {abs(real_acc - fake_acc):.2f}%")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  [[{tn:4d}, {fp:4d}]")
    logger.info(f"   [{fn:4d}, {tp:4d}]]")
    logger.info(f"{'='*70}\n")

if __name__ == "__main__":
    main()
