#!/usr/bin/env python3
"""
Real video inference for trained deepfake models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from deepfake_detector.models.efficientnet_detector import EfficientNetDeepfakeDetector
import argparse

def load_model(model_path):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetDeepfakeDetector('efficientnet_b0')
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, device

def predict_video(video_path, model, device):
    """Predict if video is real or fake"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0
    
    while frame_count < 30:  # Sample 30 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            
            face_tensor = transform(Image.fromarray(face)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(face_tensor)
                prob = torch.softmax(output, dim=1)[0][1].item()  # Fake probability
                predictions.append(prob)
        
        frame_count += 1
    
    cap.release()
    
    if predictions:
        avg_fake_prob = np.mean(predictions)
        prediction = "FAKE" if avg_fake_prob > 0.5 else "REAL"
        confidence = max(avg_fake_prob, 1 - avg_fake_prob)
        
        return prediction, confidence, avg_fake_prob
    
    return "UNKNOWN", 0.0, 0.5

def main():
    parser = argparse.ArgumentParser(description='Deepfake Video Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, device = load_model(args.model)
    
    print(f"Analyzing video: {args.video}")
    prediction, confidence, fake_prob = predict_video(args.video, model, device)
    
    print(f"\nResults:")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Fake Probability: {fake_prob:.3f}")

if __name__ == "__main__":
    main()
