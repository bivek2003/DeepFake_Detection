#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from deepfake_detector.models.efficientnet_detector import EfficientNetDeepfakeDetector
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class TestDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.face_data = []
        self._extract_faces()
    
    def _extract_faces(self):
        print("Extracting test faces...")
        for video_path, label in tqdm(zip(self.video_paths, self.labels)):
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            for idx in np.linspace(0, total-1, min(5, total), dtype=int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face = frame[y:y+h, x:x+w]
                        face = cv2.resize(face, (224, 224))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        self.face_data.append((face, label))
                        break
            cap.release()
    
    def __len__(self):
        return len(self.face_data)
    
    def __getitem__(self, idx):
        face, label = self.face_data[idx]
        return self.transform(Image.fromarray(face)), torch.tensor(label, dtype=torch.long)

# Load test videos
from sklearn.model_selection import train_test_split
import glob

real_videos = glob.glob('datasets/original_sequences/*/c23/videos/*.mp4')
fake_videos = glob.glob('datasets/manipulated_sequences/*/c23/videos/*.mp4')

all_videos = real_videos + fake_videos
all_labels = [0]*len(real_videos) + [1]*len(fake_videos)

_, test_videos, _, test_labels = train_test_split(all_videos, all_labels, test_size=0.15, random_state=42, stratify=all_labels)

print(f"Test set: {len(test_videos)} videos ({test_labels.count(0)} real, {test_labels.count(1)} fake)")

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientNetDeepfakeDetector('efficientnet_b0').to(device)
checkpoint = torch.load('models/faceforensics_improved.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create test dataset
test_dataset = TestDataset(test_videos, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Evaluate
predictions = []
targets = []

print("Evaluating on test set...")
with torch.no_grad():
    for data, target in tqdm(test_loader):
        data = data.to(device)
        output = model(data)
        _, pred = torch.max(output, 1)
        predictions.extend(pred.cpu().numpy())
        targets.extend(target.numpy())

# Calculate metrics
test_acc = accuracy_score(targets, predictions) * 100
test_f1 = f1_score(targets, predictions, average='binary')
cm = confusion_matrix(targets, predictions)

print("\n" + "="*60)
print("FINAL TEST SET EVALUATION")
print("="*60)
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(targets, predictions, target_names=['Real', 'Fake']))

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title(f'Test Set Performance: {test_acc:.2f}% Accuracy')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('results/final_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved to results/final_test_confusion_matrix.png")

if test_acc >= 87:
    print("\n✅ SUCCESS: Model achieves production target (≥87%)")
    print("Phase 2 COMPLETE - Ready for Phase 3!")
else:
    print(f"\n⚠️  Test accuracy: {test_acc:.2f}% (Target: 87%)")
