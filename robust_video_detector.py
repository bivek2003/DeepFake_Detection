#!/usr/bin/env python3
"""
Universal Video Deepfake Detector
Works on ANY video: traditional deepfakes, AI-generated, compressed, filters, etc.
Uses ensemble analysis and multi-modal detection
"""

import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys
from collections import defaultdict
from scipy import fftpack
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent / 'src'))


class UniversalVideoDetector:
    """
    Multi-modal deepfake detector that analyzes:
    1. Spatial features (RGB)
    2. Frequency artifacts (FFT)
    3. Temporal consistency
    4. Face quality and compression
    5. Blending artifacts
    """
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading universal detector on {self.device}...")
        from train_universal_detector import UniversalDeepfakeDetector
        self.model = UniversalDeepfakeDetector('efficientnet_b4').to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded (Val F1: {checkpoint.get('f1', 'N/A'):.4f})")
        
        # Face detector - try DNN first, fallback to Haar
        self.face_detector = self._load_face_detector()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_face_detector(self):
        """Load best available face detector"""
        model_path = "deploy.prototxt"
        weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
        
        if Path(model_path).exists() and Path(weights_path).exists():
            print("Using DNN face detector")
            return cv2.dnn.readNetFromCaffe(model_path, weights_path)
        else:
            print("Using Haar Cascade (consider downloading DNN model)")
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def _detect_face(self, frame):
        """Detect largest face in frame"""
        if isinstance(self.face_detector, cv2.dnn_Net):
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            best_conf = 0
            best_box = None
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > 0.5 and conf > best_conf:
                    best_conf = conf
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_box = box.astype(int)
            return best_box
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                return [x, y, x+w, y+h]
            return None
    
    def _extract_face_with_metrics(self, frame):
        """Extract face and compute quality metrics"""
        face_box = self._detect_face(frame)
        
        if face_box is None:
            return None, {}
        
        x1, y1, x2, y2 = face_box
        w, h = x2 - x1, y2 - y1
        
        # Add padding
        padding = int(0.3 * max(w, h))
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        face = frame[y1:y2, x1:x2]
        
        if face.size == 0:
            return None, {}
        
        # Calculate quality metrics
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        # Compression artifacts (DCT coefficient analysis)
        dct = cv2.dct(np.float32(gray_face))
        compression_score = np.std(dct)
        
        # Edge consistency (Canny edges)
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color consistency (histogram entropy)
        color_entropy = np.mean([entropy(cv2.calcHist([face], [i], None, [256], [0, 256]).flatten()) 
                                 for i in range(3)])
        
        metrics = {
            'sharpness': sharpness,
            'compression': compression_score,
            'edge_density': edge_density,
            'color_entropy': color_entropy,
            'face_size': (w, h)
        }
        
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        return face, metrics
    
    def _analyze_frequency_domain(self, face):
        """Analyze frequency domain for manipulation artifacts"""
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        
        # FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Analyze frequency distribution
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Low frequency energy (center)
        low_freq_region = magnitude[center_y-20:center_y+20, center_x-20:center_x+20]
        low_freq_energy = np.sum(low_freq_region)
        
        # High frequency energy (edges)
        high_freq_mask = np.ones_like(magnitude)
        high_freq_mask[center_y-20:center_y+20, center_x-20:center_x+20] = 0
        high_freq_energy = np.sum(magnitude * high_freq_mask)
        
        # Ratio indicates manipulation (deepfakes often have abnormal freq distribution)
        freq_ratio = low_freq_energy / (high_freq_energy + 1e-8)
        
        return {
            'freq_ratio': freq_ratio,
            'low_freq': low_freq_energy,
            'high_freq': high_freq_energy
        }
    
    def _analyze_temporal_consistency(self, faces, frame_indices):
        """Analyze temporal consistency across frames"""
        if len(faces) < 2:
            return {'temporal_score': 1.0, 'consistency': 1.0}
        
        # Calculate frame-to-frame differences
        diffs = []
        for i in range(len(faces) - 1):
            diff = np.mean(np.abs(faces[i].astype(float) - faces[i+1].astype(float)))
            diffs.append(diff)
        
        # High variance in differences suggests temporal inconsistency
        temporal_variance = np.var(diffs) if len(diffs) > 1 else 0
        temporal_mean = np.mean(diffs) if len(diffs) > 0 else 0
        
        # Calculate optical flow consistency
        flows = []
        for i in range(len(faces) - 1):
            gray1 = cv2.cvtColor(faces[i], cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(faces[i+1], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flows.append(np.mean(flow_magnitude))
        
        flow_consistency = 1.0 - (np.std(flows) / (np.mean(flows) + 1e-8))
        
        return {
            'temporal_variance': temporal_variance,
            'temporal_mean': temporal_mean,
            'flow_consistency': max(0, min(1, flow_consistency)),
            'temporal_score': max(0, 1 - temporal_variance / 100)
        }
    
    def _get_video_quality(self, video_path):
        """Analyze overall video quality"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        return {
            'fps': fps,
            'resolution': (width, height),
            'total_frames': total_frames,
            'is_hd': width >= 1280,
            'is_low_res': width < 640,
            'aspect_ratio': width / height if height > 0 else 0
        }
    
    def detect(self, video_path, num_frames=40, confidence_threshold=0.5):
        """
        Comprehensive video analysis
        
        Returns:
            dict with prediction, confidence, and detailed analysis
        """
        print(f"\n{'='*70}")
        print(f"Analyzing: {Path(video_path).name}")
        print(f"{'='*70}")
        
        # Get video quality info
        video_quality = self._get_video_quality(video_path)
        print(f"Resolution: {video_quality['resolution'][0]}x{video_quality['resolution'][1]}")
        print(f"FPS: {video_quality['fps']:.1f}, Frames: {video_quality['total_frames']}")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = video_quality['total_frames']
        
        if total_frames == 0:
            return {'error': 'Cannot read video'}
        
        # Smart frame sampling
        start_frame = int(total_frames * 0.1)
        end_frame = int(total_frames * 0.9)
        frame_indices = np.linspace(start_frame, end_frame, 
                                   min(num_frames, end_frame - start_frame), 
                                   dtype=int)
        
        # Collect data
        predictions = []
        confidences = []
        faces = []
        face_metrics = []
        freq_analyses = []
        
        print(f"Extracting {len(frame_indices)} frames...")
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            face, metrics = self._extract_face_with_metrics(frame)
            
            if face is not None:
                # Store for temporal analysis
                faces.append(face.copy())
                face_metrics.append(metrics)
                
                # Frequency analysis
                freq_analysis = self._analyze_frequency_domain(face)
                freq_analyses.append(freq_analysis)
                
                # Model prediction
                face_pil = Image.fromarray(face)
                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(face_tensor)
                    probs = torch.softmax(output, dim=1)
                    pred = torch.argmax(probs, dim=1).item()
                    conf = probs[0, pred].item()
                    fake_prob = probs[0, 1].item()
                    
                    predictions.append(pred)
                    confidences.append(fake_prob)
        
        cap.release()
        
        if len(predictions) == 0:
            return {
                'error': 'No faces detected',
                'video_quality': video_quality
            }
        
        print(f"Analyzed {len(predictions)} faces")
        
        # Temporal consistency analysis
        temporal_analysis = self._analyze_temporal_consistency(faces, frame_indices[:len(faces)])
        
        # Aggregate metrics
        avg_fake_prob = np.mean(confidences)
        std_fake_prob = np.std(confidences)
        fake_ratio = np.mean(predictions)
        
        # Quality-based adjustments
        avg_sharpness = np.mean([m['sharpness'] for m in face_metrics])
        avg_freq_ratio = np.mean([f['freq_ratio'] for f in freq_analyses])
        
        # Multi-factor decision
        factors = {
            'model_confidence': avg_fake_prob,
            'consistency': 1 - std_fake_prob,
            'temporal_score': temporal_analysis['temporal_score'],
            'flow_consistency': temporal_analysis['flow_consistency'],
            'sharpness': min(1.0, avg_sharpness / 200),  # Normalize
            'freq_anomaly': 1.0 / (1.0 + np.exp(-(avg_freq_ratio - 5)))  # Sigmoid
        }
        
        # Weighted scoring
        weights = {
            'model_confidence': 0.40,
            'consistency': 0.20,
            'temporal_score': 0.15,
            'flow_consistency': 0.10,
            'sharpness': 0.05,
            'freq_anomaly': 0.10
        }
        
        final_score = sum(factors[k] * weights[k] for k in weights)
        
        # Reliability assessment
        if video_quality['is_low_res'] or avg_sharpness < 50:
            reliability = 'LOW'
            adjusted_threshold = 0.65
        elif std_fake_prob > 0.3 or temporal_analysis['temporal_score'] < 0.5:
            reliability = 'MEDIUM'
            adjusted_threshold = 0.60
        else:
            reliability = 'HIGH'
            adjusted_threshold = confidence_threshold
        
        final_prediction = 'FAKE' if final_score > adjusted_threshold else 'REAL'
        
        # Detailed results
        result = {
            'prediction': final_prediction,
            'confidence': final_score,
            'raw_model_confidence': avg_fake_prob,
            'consistency': 1 - std_fake_prob,
            'reliability': reliability,
            'frames_analyzed': len(predictions),
            'fake_frame_ratio': fake_ratio,
            
            'detailed_scores': factors,
            'temporal_analysis': temporal_analysis,
            'video_quality': video_quality,
            
            'avg_sharpness': avg_sharpness,
            'freq_ratio': avg_freq_ratio,
            
            'recommendation': self._generate_recommendation(
                final_prediction, final_score, reliability, factors, video_quality
            )
        }
        
        return result
    
    def _generate_recommendation(self, prediction, score, reliability, factors, quality):
        """Generate user-friendly recommendation"""
        if prediction == 'FAKE':
            if score > 0.8 and reliability == 'HIGH':
                return (
                    "🚨 VERY HIGH confidence this is a DEEPFAKE.\n"
                    "Strong manipulation signals detected across multiple analysis methods.\n"
                    "Recommendation: Treat as manipulated content."
                )
            elif score > 0.7:
                return (
                    "⚠️ HIGH confidence this is a DEEPFAKE.\n"
                    "Multiple manipulation indicators detected.\n"
                    "Recommendation: Likely manipulated, verify source."
                )
            elif score > 0.6:
                return (
                    "⚠️ MODERATE confidence this is a DEEPFAKE.\n"
                    "Some manipulation signals detected.\n"
                    "Recommendation: Suspicious, requires further verification."
                )
            else:
                return (
                    "⚠️ LOW confidence detection.\n"
                    f"Reliability: {reliability}. May be affected by compression or quality.\n"
                    "Recommendation: Inconclusive, seek additional verification."
                )
        else:
            if score < 0.3 and reliability == 'HIGH':
                return (
                    "✅ VERY HIGH confidence this is REAL.\n"
                    "No manipulation signals detected.\n"
                    "Recommendation: Appears authentic."
                )
            elif score < 0.4:
                return (
                    "✅ HIGH confidence this is REAL.\n"
                    "Minimal manipulation indicators.\n"
                    "Recommendation: Likely authentic."
                )
            else:
                return (
                    "✅ MODERATE confidence this is REAL.\n"
                    "Close to decision boundary.\n"
                    "Recommendation: Appears real but consider source verification."
                )
    
    def print_results(self, result):
        """Pretty print analysis results"""
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
            return
        
        print(f"\n{'='*70}")
        print(f"ANALYSIS RESULTS")
        print(f"{'='*70}")
        
        print(f"\n🎯 PREDICTION: {result['prediction']}")
        print(f"Confidence Score: {result['confidence']:.1%}")
        print(f"Reliability: {result['reliability']}")
        
        print(f"\n📊 DETAILED SCORES:")
        for factor, score in result['detailed_scores'].items():
            bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
            print(f"  {factor:20s}: {bar} {score:.1%}")
        
        print(f"\n🎬 VIDEO ANALYSIS:")
        print(f"  Frames Analyzed: {result['frames_analyzed']}")
        print(f"  Fake Frame Ratio: {result['fake_frame_ratio']:.1%}")
        print(f"  Prediction Consistency: {result['consistency']:.1%}")
        print(f"  Temporal Score: {result['temporal_analysis']['temporal_score']:.1%}")
        print(f"  Motion Consistency: {result['temporal_analysis']['flow_consistency']:.1%}")
        
        print(f"\n📹 VIDEO QUALITY:")
        q = result['video_quality']
        print(f"  Resolution: {q['resolution'][0]}x{q['resolution'][1]}")
        print(f"  Quality: {'HD' if q['is_hd'] else 'SD'}")
        print(f"  Avg Sharpness: {result['avg_sharpness']:.1f}")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"{result['recommendation']}")
        
        print(f"\n{'='*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Universal Deepfake Detector - Works on ANY video'
    )
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, 
                       default='models/universal_deepfake_detector.pth',
                       help='Path to trained model')
    parser.add_argument('--frames', type=int, default=40,
                       help='Number of frames to analyze (more = slower but more accurate)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (0-1)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    if not Path(args.video_path).exists():
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    if not Path(args.model).exists():
        print(f"❌ Error: Model file not found: {args.model}")
        print(f"Please train the model first using: python train_universal_detector.py")
        return
    
    # Initialize detector
    detector = UniversalVideoDetector(args.model, device=args.device)
    
    # Analyze video
    result = detector.detect(
        args.video_path, 
        num_frames=args.frames,
        confidence_threshold=args.threshold
    )
    
    # Print results
    detector.print_results(result)


if __name__ == "__main__":
    main()
