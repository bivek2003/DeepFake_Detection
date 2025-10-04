"""
Inference service for deepfake detection - Updated for robust model
"""

import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import logging
from typing import Dict
import tempfile
import os

logger = logging.getLogger(__name__)

class InferenceService:
    """Service for running deepfake detection inference"""
    
    def __init__(self, model_path: str, device: str = 'cuda', confidence_threshold: float = 0.65):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()
        logger.info(f"InferenceService initialized on {self.device}")
    
    def _load_model(self):
        """Load the trained model"""
        from deepfake_detector.models.efficientnet_detector import EfficientNetDeepfakeDetector
        
        self.model = EfficientNetDeepfakeDetector('efficientnet_b0')
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {self.model_path}")
        logger.info(f"Model accuracy: {checkpoint.get('accuracy', 'N/A')}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    def _extract_face(self, frame: np.ndarray) -> np.ndarray:
        """Extract face from frame"""
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
        
        return cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB)
    
    def _get_prediction_label(self, fake_prob: float) -> tuple:
        """Get prediction label with uncertainty handling"""
        if fake_prob > self.confidence_threshold:
            return "fake", True
        elif fake_prob < (1 - self.confidence_threshold):
            return "real", True
        else:
            # Uncertain range
            return "fake" if fake_prob > 0.5 else "real", False
    
    def predict_image(self, image: np.ndarray) -> Dict:
        """Predict if image is real or fake"""
        try:
            face = self._extract_face(image)
            face_pil = Image.fromarray(face)
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(face_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                fake_prob = probabilities[1].item()
                
            prediction, is_confident = self._get_prediction_label(fake_prob)
            confidence = max(fake_prob, 1 - fake_prob)
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "fake_probability": float(fake_prob),
                "real_probability": float(1 - fake_prob),
                "is_confident": is_confident,
                "warning": None if is_confident else "Low confidence - manual review recommended"
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_video(self, video_path: str, max_frames: int = 30) -> Dict:
        """Predict if video is real or fake"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            predictions = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    result = self.predict_image(frame)
                    predictions.append(result['fake_probability'])
            
            cap.release()
            
            if not predictions:
                raise ValueError("No faces detected in video")
            
            avg_fake_prob = np.mean(predictions)
            prediction, is_confident = self._get_prediction_label(avg_fake_prob)
            confidence = max(avg_fake_prob, 1 - avg_fake_prob)
            
            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "fake_probability": float(avg_fake_prob),
                "real_probability": float(1 - avg_fake_prob),
                "frames_analyzed": len(predictions),
                "total_frames": total_frames,
                "is_confident": is_confident,
                "warning": None if is_confident else "Low confidence - video may be outside training distribution"
            }
        
        except Exception as e:
            logger.error(f"Video prediction error: {e}")
            raise
        finally:
            if 'cap' in locals():
                cap.release()
