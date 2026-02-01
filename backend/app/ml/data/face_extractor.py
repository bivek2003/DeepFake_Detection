"""
Face extraction module for deepfake detection training.

Uses MTCNN for accurate face detection and alignment.
Extracts faces from videos and saves them as images for training.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
from tqdm import tqdm
import torch
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not installed. Using OpenCV face detection.")


@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    landmarks: Optional[np.ndarray] = None


class FaceExtractor:
    """
    Extract and align faces from videos for deepfake detection training.
    
    Uses MTCNN for face detection with optional OpenCV fallback.
    """
    
    def __init__(
        self,
        output_size: int = 380,
        margin: float = 0.3,
        min_face_size: int = 60,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize face extractor.
        
        Args:
            output_size: Size of extracted face images
            margin: Margin around detected face (percentage)
            min_face_size: Minimum face size to detect
            device: Device for MTCNN
            batch_size: Batch size for video processing
        """
        self.output_size = output_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.device = device
        self.batch_size = batch_size
        
        # Initialize MTCNN
        if MTCNN_AVAILABLE:
            self.detector = MTCNN(
                image_size=output_size,
                margin=int(output_size * margin),
                min_face_size=min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,
                device=device,
                keep_all=False,  # Only keep largest face
            )
            print(f"Using MTCNN face detector on {device}")
        else:
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade_path)
            print("Using OpenCV Haar Cascade face detector")
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: BGR image as numpy array
        
        Returns:
            List of FaceDetection objects
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if MTCNN_AVAILABLE:
            return self._detect_mtcnn(rgb_image)
        else:
            return self._detect_opencv(image)
    
    def _detect_mtcnn(self, rgb_image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using MTCNN."""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Detect faces
            boxes, probs, landmarks = self.detector.detect(pil_image, landmarks=True)
            
            if boxes is None:
                return []
            
            detections = []
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob < 0.9:  # Confidence threshold
                    continue
                
                x1, y1, x2, y2 = [int(b) for b in box]
                landmark = landmarks[i] if landmarks is not None else None
                
                detections.append(FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(prob),
                    landmarks=landmark
                ))
            
            return detections
            
        except Exception as e:
            print(f"MTCNN detection error: {e}")
            return []
    
    def _detect_opencv(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append(FaceDetection(
                bbox=(x, y, x + w, y + h),
                confidence=1.0,
                landmarks=None
            ))
        
        return detections
    
    def extract_face(
        self,
        image: np.ndarray,
        detection: Optional[FaceDetection] = None
    ) -> Optional[np.ndarray]:
        """
        Extract and align face from image.
        
        Args:
            image: BGR image
            detection: Optional pre-computed detection
        
        Returns:
            Extracted face image or None
        """
        if detection is None:
            detections = self.detect_faces(image)
            if not detections:
                return None
            detection = detections[0]  # Use first (largest) face
        
        x1, y1, x2, y2 = detection.bbox
        h, w = image.shape[:2]
        
        # Add margin
        face_w = x2 - x1
        face_h = y2 - y1
        margin_x = int(face_w * self.margin)
        margin_y = int(face_h * self.margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Extract and resize
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        face = cv2.resize(face, (self.output_size, self.output_size))
        
        return face
    
    def extract_from_video(
        self,
        video_path: str,
        output_dir: str,
        num_frames: int = 32,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract faces from a video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted faces
            num_frames: Number of frames to extract
            skip_existing: Skip if output already exists
        
        Returns:
            Dictionary with extraction results
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = video_path.stem
        
        # Check if already extracted
        if skip_existing:
            existing = list(output_dir.glob(f"{video_name}_*.jpg"))
            if len(existing) >= num_frames // 2:  # At least half extracted
                return {
                    "video": str(video_path),
                    "status": "skipped",
                    "faces": len(existing)
                }
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {
                "video": str(video_path),
                "status": "error",
                "error": "Could not open video"
            }
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return {
                "video": str(video_path),
                "status": "error",
                "error": "Video has no frames"
            }
        
        # Calculate frame indices to sample
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        extracted = 0
        failed = 0
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                failed += 1
                continue
            
            face = self.extract_face(frame)
            
            if face is not None:
                output_path = output_dir / f"{video_name}_{i:04d}.jpg"
                cv2.imwrite(str(output_path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted += 1
            else:
                failed += 1
        
        cap.release()
        
        return {
            "video": str(video_path),
            "status": "success",
            "faces": extracted,
            "failed": failed
        }
    
    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        num_frames: int = 32,
        num_workers: int = 4,
    ) -> Dict[str, Any]:
        """
        Process entire dataset directory.
        
        Args:
            input_dir: Directory containing videos
            output_dir: Directory to save extracted faces
            num_frames: Frames per video
            num_workers: Not used (for API compatibility)
        
        Returns:
            Processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Find all videos
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        videos = []
        for ext in video_extensions:
            videos.extend(input_dir.rglob(f"*{ext}"))
        
        print(f"Found {len(videos)} videos in {input_dir}")
        
        results = {
            "total": len(videos),
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "total_faces": 0,
        }
        
        for video_path in tqdm(videos, desc="Extracting faces"):
            # Preserve directory structure
            rel_path = video_path.relative_to(input_dir)
            video_output_dir = output_dir / rel_path.parent / video_path.stem
            
            result = self.extract_from_video(
                str(video_path),
                str(video_output_dir),
                num_frames=num_frames
            )
            
            if result["status"] == "success":
                results["success"] += 1
                results["total_faces"] += result["faces"]
            elif result["status"] == "skipped":
                results["skipped"] += 1
                results["total_faces"] += result["faces"]
            else:
                results["failed"] += 1
                print(f"Failed: {video_path} - {result.get('error', 'Unknown error')}")
        
        return results


def main():
    """CLI for face extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract faces from videos")
    parser.add_argument("--input", type=str, required=True, help="Input video or directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--num-frames", type=int, default=32, help="Frames per video")
    parser.add_argument("--size", type=int, default=380, help="Output face size")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    extractor = FaceExtractor(
        output_size=args.size,
        device=args.device if torch.cuda.is_available() else "cpu"
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = extractor.extract_from_video(
            str(input_path),
            args.output,
            num_frames=args.num_frames
        )
        print(f"Result: {result}")
    else:
        results = extractor.process_dataset(
            args.input,
            args.output,
            num_frames=args.num_frames
        )
        print(f"\nProcessing complete:")
        print(f"  Total videos: {results['total']}")
        print(f"  Successful: {results['success']}")
        print(f"  Skipped: {results['skipped']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Total faces extracted: {results['total_faces']}")


if __name__ == "__main__":
    main()
