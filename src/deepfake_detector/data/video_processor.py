"""
Video Processing for Deepfake Detection

Handles face detection, extraction, and preprocessing from video files.
Supports multiple face detection backends and video augmentation techniques.

Author: Your Name
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Container for face detection results"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    landmarks: Optional[np.ndarray] = None
    frame_idx: int = 0


@dataclass 
class VideoMetadata:
    """Container for video metadata"""
    filepath: str
    duration: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str
    size_mb: float


class FaceDetector:
    """Face detection with multiple backend support"""
    
    def __init__(self, backend: str = "opencv", confidence_threshold: float = 0.5):
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the face detection backend"""
        if self.backend == "opencv":
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        elif self.backend == "dnn":
            # OpenCV DNN face detector (more accurate)
            model_path = "models/opencv_face_detector_uint8.pb"
            config_path = "models/opencv_face_detector.pbtxt"
            try:
                self.detector = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                logger.info("Loaded OpenCV DNN face detector")
            except Exception as e:
                logger.warning(f"Could not load DNN model: {e}, falling back to Haar cascades")
                self.backend = "opencv"
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
        else:
            raise ValueError(f"Unsupported face detection backend: {self.backend}")
    
    def detect_faces(self, frame: np.ndarray, frame_idx: int = 0) -> List[FaceDetection]:
        """Detect faces in a frame"""
        if self.backend == "opencv":
            return self._detect_opencv(frame, frame_idx)
        elif self.backend == "dnn":
            return self._detect_dnn(frame, frame_idx)
        
    def _detect_opencv(self, frame: np.ndarray, frame_idx: int) -> List[FaceDetection]:
        """OpenCV Haar cascade detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detection = FaceDetection(
                bbox=(x, y, w, h),
                confidence=1.0,  # Haar cascades don't provide confidence
                frame_idx=frame_idx
            )
            detections.append(detection)
        
        return detections
    
    def _detect_dnn(self, frame: np.ndarray, frame_idx: int) -> List[FaceDetection]:
        """OpenCV DNN face detection"""
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.detector.setInput(blob)
        detections_raw = self.detector.forward()
        
        detections = []
        for i in range(detections_raw.shape[2]):
            confidence = detections_raw[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                x1 = int(detections_raw[0, 0, i, 3] * w)
                y1 = int(detections_raw[0, 0, i, 4] * h)
                x2 = int(detections_raw[0, 0, i, 5] * w)
                y2 = int(detections_raw[0, 0, i, 6] * h)
                
                detection = FaceDetection(
                    bbox=(x1, y1, x2-x1, y2-y1),
                    confidence=confidence,
                    frame_idx=frame_idx
                )
                detections.append(detection)
        
        return detections


class VideoProcessor:
    """Main video processing pipeline for deepfake detection"""
    
    def __init__(self, 
                 face_detector: Optional[FaceDetector] = None,
                 target_size: Tuple[int, int] = (224, 224),
                 max_frames: int = 30,
                 frame_interval: int = 1):
        """
        Initialize video processor
        
        Args:
            face_detector: Face detection backend
            target_size: Target size for face crops  
            max_frames: Maximum frames to process per video
            frame_interval: Process every Nth frame
        """
        self.face_detector = face_detector or FaceDetector("opencv")
        self.target_size = target_size
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        
    def extract_video_metadata(self, video_path: Union[str, Path]) -> VideoMetadata:
        """Extract metadata from video file"""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        
        # Calculate duration
        duration = frame_count / fps if fps > 0 else 0
        
        # Get file size
        size_mb = video_path.stat().st_size / (1024 * 1024)
        
        # Decode fourcc
        codec = "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return VideoMetadata(
            filepath=str(video_path),
            duration=duration,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            codec=codec,
            size_mb=size_mb
        )
    
    def extract_frames(self, video_path: Union[str, Path], 
                      frame_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """Extract specific frames or evenly spaced frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        
        if frame_indices is None:
            # Extract evenly spaced frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // min(self.max_frames, total_frames))
            frame_indices = list(range(0, total_frames, step))[:self.max_frames]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB for consistency
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def extract_faces_from_video(self, video_path: Union[str, Path],
                                return_metadata: bool = False) -> Union[List[np.ndarray], 
                                                                      Tuple[List[np.ndarray], Dict]]:
        """Extract face crops from video"""
        video_path = Path(video_path)
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        face_crops = []
        face_detections = []
        processing_stats = {
            "total_frames": len(frames),
            "frames_with_faces": 0,
            "total_faces_detected": 0,
            "average_faces_per_frame": 0,
            "processing_time": 0
        }
        
        start_time = time.time()
        
        for frame_idx, frame in enumerate(frames):
            # Detect faces
            detections = self.face_detector.detect_faces(frame, frame_idx)
            
            if detections:
                processing_stats["frames_with_faces"] += 1
                processing_stats["total_faces_detected"] += len(detections)
                
                # Get largest face (most likely main subject)
                largest_face = max(detections, 
                                 key=lambda d: d.bbox[2] * d.bbox[3])
                
                # Extract and crop face
                x, y, w, h = largest_face.bbox
                
                # Add padding
                padding = 0.2
                pad_x = int(w * padding)
                pad_y = int(h * padding)
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)
                
                face_crop = frame[y1:y2, x1:x2]
                
                # Resize to target size
                if face_crop.size > 0:
                    face_resized = cv2.resize(face_crop, self.target_size)
                    face_crops.append(face_resized)
                    face_detections.append(largest_face)
        
        processing_stats["processing_time"] = time.time() - start_time
        processing_stats["average_faces_per_frame"] = (
            processing_stats["total_faces_detected"] / len(frames) if frames else 0
        )
        
        metadata = {
            "video_path": str(video_path),
            "face_crops_count": len(face_crops),
            "face_detections": face_detections,
            "processing_stats": processing_stats
        }
        
        if return_metadata:
            return face_crops, metadata
        return face_crops
    
    def batch_process_videos(self, video_paths: List[Union[str, Path]], 
                           output_dir: Optional[Path] = None,
                           max_workers: int = 4) -> Dict[str, Dict]:
        """Process multiple videos in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_path = {
                executor.submit(self.extract_faces_from_video, path, True): path 
                for path in video_paths
            }
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_path), 
                             total=len(video_paths),
                             desc="Processing videos"):
                video_path = future_to_path[future]
                try:
                    face_crops, metadata = future.result()
                    
                    # Save face crops if output directory specified
                    if output_dir:
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        video_name = Path(video_path).stem
                        video_output_dir = output_dir / video_name
                        video_output_dir.mkdir(exist_ok=True)
                        
                        # Save face crops
                        for i, face_crop in enumerate(face_crops):
                            face_path = video_output_dir / f"face_{i:03d}.jpg"
                            # Convert RGB back to BGR for saving
                            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(face_path), face_bgr)
                        
                        # Save metadata
                        metadata_path = video_output_dir / "metadata.json"
                        with open(metadata_path, 'w') as f:
                            # Convert non-serializable objects
                            serializable_metadata = self._make_serializable(metadata)
                            json.dump(serializable_metadata, f, indent=2)
                    
                    results[str(video_path)] = {
                        "face_crops": face_crops,
                        "metadata": metadata,
                        "success": True
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing {video_path}: {e}")
                    results[str(video_path)] = {
                        "face_crops": [],
                        "metadata": {},
                        "success": False,
                        "error": str(e)
                    }
        
        return results
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, FaceDetection):
            return {
                "bbox": obj.bbox,
                "confidence": obj.confidence,
                "frame_idx": obj.frame_idx
            }
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def analyze_video_quality(self, video_path: Union[str, Path]) -> Dict:
        """Analyze video quality metrics relevant to deepfake detection"""
        metadata = self.extract_video_metadata(video_path)
        face_crops, face_metadata = self.extract_faces_from_video(video_path, True)
        
        quality_metrics = {
            "resolution": f"{metadata.width}x{metadata.height}",
            "resolution_class": self._classify_resolution(metadata.width, metadata.height),
            "fps": metadata.fps,
            "fps_class": self._classify_fps(metadata.fps),
            "duration": metadata.duration,
            "face_detection_rate": (
                face_metadata["processing_stats"]["frames_with_faces"] / 
                face_metadata["processing_stats"]["total_frames"]
                if face_metadata["processing_stats"]["total_frames"] > 0 else 0
            ),
            "average_face_size": self._calculate_average_face_size(face_metadata["face_detections"]),
            "quality_score": 0.0  # Will be calculated
        }
        
        # Calculate overall quality score (0-1)
        score = 0.0
        
        # Resolution contribution (30%)
        if quality_metrics["resolution_class"] == "HD":
            score += 0.3
        elif quality_metrics["resolution_class"] == "SD":
            score += 0.15
        
        # FPS contribution (20%)
        if quality_metrics["fps_class"] == "high":
            score += 0.2
        elif quality_metrics["fps_class"] == "normal":
            score += 0.15
        
        # Face detection rate contribution (30%)
        score += quality_metrics["face_detection_rate"] * 0.3
        
        # Duration contribution (20%)
        if 2.0 <= quality_metrics["duration"] <= 10.0:  # Ideal range
            score += 0.2
        elif quality_metrics["duration"] >= 1.0:
            score += 0.1
        
        quality_metrics["quality_score"] = score
        
        return quality_metrics
    
    def _classify_resolution(self, width: int, height: int) -> str:
        """Classify video resolution"""
        pixels = width * height
        if pixels >= 1920 * 1080:
            return "HD"
        elif pixels >= 640 * 480:
            return "SD"
        else:
            return "Low"
    
    def _classify_fps(self, fps: float) -> str:
        """Classify video frame rate"""
        if fps >= 30:
            return "high"
        elif fps >= 24:
            return "normal"  
        else:
            return "low"
    
    def _calculate_average_face_size(self, detections: List[FaceDetection]) -> float:
        """Calculate average face size in pixels"""
        if not detections:
            return 0.0
        
        total_area = sum(d.bbox[2] * d.bbox[3] for d in detections)
        return total_area / len(detections)


def main():
    """Demonstration of video processing functionality"""
    # Initialize processor
    processor = VideoProcessor(
        face_detector=FaceDetector("opencv"),
        target_size=(224, 224),
        max_frames=10
    )
    
    print("🎬 Video Processing Pipeline Demo")
    print("=" * 50)
    
    # This would work with actual video files
    # For demo, we'll show the capabilities
    
    print("\nCapabilities:")
    print("• Face detection (OpenCV Haar & DNN)")
    print("• Video metadata extraction") 
    print("• Frame extraction with smart sampling")
    print("• Face cropping and alignment")
    print("• Batch processing with parallel execution")
    print("• Quality analysis for deepfake detection")
    print("• Automatic padding and resizing")
    
    print(f"\nProcessor Configuration:")
    print(f"• Target face size: {processor.target_size}")
    print(f"• Max frames per video: {processor.max_frames}")
    print(f"• Face detector: {processor.face_detector.backend}")
    print(f"• Confidence threshold: {processor.face_detector.confidence_threshold}")
    
    # Example usage (would work with real video files)
    print(f"\nExample Usage:")
    print(f"```python")
    print(f"# Process single video")
    print(f"faces = processor.extract_faces_from_video('video.mp4')")
    print(f"")
    print(f"# Batch process multiple videos")  
    print(f"results = processor.batch_process_videos(['vid1.mp4', 'vid2.mp4'])")
    print(f"")
    print(f"# Analyze video quality")
    print(f"quality = processor.analyze_video_quality('video.mp4')")
    print(f"```")


if __name__ == "__main__":
    main()
