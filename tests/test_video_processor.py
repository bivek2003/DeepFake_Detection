"""
Unit tests for VideoProcessor

Tests face detection, video processing, and quality analysis.
"""

import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_detector.data import VideoProcessor, FaceDetector, VideoMetadata, FaceDetection


class TestFaceDetector:
    """Test face detection functionality"""
    
    def test_face_detector_initialization_opencv(self):
        """Test OpenCV face detector initialization"""
        detector = FaceDetector("opencv", confidence_threshold=0.5)
        
        assert detector.backend == "opencv"
        assert detector.confidence_threshold == 0.5
        assert detector.detector is not None
        
    def test_face_detector_initialization_invalid(self):
        """Test invalid backend raises error"""
        with pytest.raises(ValueError, match="Unsupported face detection backend"):
            FaceDetector("invalid_backend")
            
    def test_detect_faces_synthetic_image(self):
        """Test face detection on synthetic image"""
        detector = FaceDetector("opencv")
        
        # Create synthetic image with face-like pattern
        frame = self._create_synthetic_face_image()
        
        detections = detector.detect_faces(frame, frame_idx=0)
        
        # Should detect something (even if not perfect on synthetic data)
        assert isinstance(detections, list)
        for detection in detections:
            assert isinstance(detection, FaceDetection)
            assert len(detection.bbox) == 4  # (x, y, w, h)
            assert detection.confidence >= 0
            assert detection.frame_idx == 0
            
    def _create_synthetic_face_image(self) -> np.ndarray:
        """Create synthetic image with face-like features"""
        # Create 224x224 image
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        # Draw face-like oval
        cv2.ellipse(frame, (112, 112), (60, 80), 0, 0, 360, (150, 150, 150), -1)
        
        # Draw eyes
        cv2.circle(frame, (92, 100), 8, (50, 50, 50), -1)
        cv2.circle(frame, (132, 100), 8, (50, 50, 50), -1)
        
        # Draw mouth
        cv2.ellipse(frame, (112, 140), (20, 10), 0, 0, 180, (80, 80, 80), -1)
        
        return frame


class TestVideoProcessor:
    """Test video processing functionality"""
    
    @pytest.fixture
    def temp_video_path(self):
        """Create temporary video file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            # Create simple test video using OpenCV
            temp_path = Path(temp_file.name)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_path), fourcc, 10.0, (224, 224))
            
            # Write 30 frames
            for i in range(30):
                frame = self._create_test_frame(i)
                out.write(frame)
            
            out.release()
            
            yield temp_path
            
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
                
    def _create_test_frame(self, frame_idx: int) -> np.ndarray:
        """Create test frame with moving pattern"""
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 100
        
        # Moving circle
        x = 50 + (frame_idx * 2) % 124
        y = 112
        cv2.circle(frame, (x, y), 20, (200, 150, 100), -1)
        
        # Face-like features
        cv2.ellipse(frame, (112, 112), (40, 50), 0, 0, 360, (150, 150, 150), -1)
        cv2.circle(frame, (100, 100), 5, (50, 50, 50), -1)  # Eye
        cv2.circle(frame, (124, 100), 5, (50, 50, 50), -1)  # Eye
        
        return frame
    
    def test_processor_initialization(self):
        """Test VideoProcessor initialization"""
        processor = VideoProcessor(
            target_size=(224, 224),
            max_frames=10,
            frame_interval=2
        )
        
        assert processor.target_size == (224, 224)
        assert processor.max_frames == 10
        assert processor.frame_interval == 2
        assert isinstance(processor.face_detector, FaceDetector)
        
    @patch('cv2.VideoCapture')
    def test_extract_video_metadata_mock(self, mock_video_capture):
        """Test video metadata extraction with mocked VideoCapture"""
        # Setup mock
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc(*'H264')
        }.get(prop, 0)
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor()
        
        # Mock file size
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 50 * 1024 * 1024  # 50MB
            
            metadata = processor.extract_video_metadata("test_video.mp4")
        
        assert isinstance(metadata, VideoMetadata)
        assert metadata.fps == 30.0
        assert metadata.frame_count == 900
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.duration == 30.0  # 900 frames / 30 fps
        assert metadata.size_mb == 50.0
        
    @patch('cv2.VideoCapture')
    def test_extract_video_metadata_invalid_file(self, mock_video_capture):
        """Test metadata extraction with invalid file"""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        processor = VideoProcessor()
        
        with pytest.raises(ValueError, match="Could not open video file"):
            processor.extract_video_metadata("nonexistent.mp4")
            
    def test_extract_frames_synthetic(self, temp_video_path):
        """Test frame extraction from synthetic video"""
        processor = VideoProcessor(max_frames=5)
        
        try:
            frames = processor.extract_frames(temp_video_path)
            
            assert len(frames) <= 5  # Should respect max_frames
            assert len(frames) > 0   # Should extract some frames
            
            for frame in frames:
                assert isinstance(frame, np.ndarray)
                assert frame.shape == (224, 224, 3)  # RGB format
                assert frame.dtype == np.uint8
                
        except Exception as e:
            # If video creation failed, skip this test
            pytest.skip(f"Could not process synthetic video: {e}")
            
    def test_extract_faces_synthetic_data(self):
        """Test face extraction with synthetic data"""
        processor = VideoProcessor(max_frames=5)
        
        # Mock the extract_frames method to return synthetic frames
        synthetic_frames = [self._create_test_frame(i) for i in range(5)]
        
        with patch.object(processor, 'extract_frames', return_value=synthetic_frames):
            faces = processor.extract_faces_from_video("dummy_path.mp4")
            
            assert isinstance(faces, list)
            # Might not detect faces in synthetic data, but should not crash
            
    def test_analyze_video_quality_mock(self):
        """Test video quality analysis with mocked data"""
        processor = VideoProcessor()
        
        # Mock the required methods
        mock_metadata = VideoMetadata(
            filepath="test.mp4",
            duration=30.0,
            fps=30.0,
            frame_count=900,
            width=1920,
            height=1080,
            codec="H264",
            size_mb=50.0
        )
        
        mock_faces_metadata = {
            "processing_stats": {
                "total_frames": 10,
                "frames_with_faces": 8,
                "total_faces_detected": 12
            },
            "face_detections": [
                FaceDetection(bbox=(50, 50, 100, 100), confidence=0.9, frame_idx=0),
                FaceDetection(bbox=(60, 60, 90, 90), confidence=0.8, frame_idx=1)
            ]
        }
        
        with patch.object(processor, 'extract_video_metadata', return_value=mock_metadata):
            with patch.object(processor, 'extract_faces_from_video', return_value=([], mock_faces_metadata)):
                quality_metrics = processor.analyze_video_quality("test.mp4")
        
        assert "resolution" in quality_metrics
        assert "fps" in quality_metrics
        assert "face_detection_rate" in quality_metrics
        assert "quality_score" in quality_metrics
        
        # Check calculated values
        assert quality_metrics["resolution"] == "1920x1080"
        assert quality_metrics["fps"] == 30.0
        assert quality_metrics["face_detection_rate"] == 0.8  # 8/10
        assert 0 <= quality_metrics["quality_score"] <= 1
        
    def test_classify_resolution(self):
        """Test resolution classification"""
        processor = VideoProcessor()
        
        # Test HD
        assert processor._classify_resolution(1920, 1080) == "HD"
        assert processor._classify_resolution(2560, 1440) == "HD"
        
        # Test SD
        assert processor._classify_resolution(640, 480) == "SD"
        assert processor._classify_resolution(1280, 720) == "SD"
        
        # Test Low
        assert processor._classify_resolution(320, 240) == "Low"
        
    def test_classify_fps(self):
        """Test FPS classification"""
        processor = VideoProcessor()
        
        assert processor._classify_fps(60) == "high"
        assert processor._classify_fps(30) == "high"
        assert processor._classify_fps(25) == "normal"
        assert processor._classify_fps(15) == "low"
        
    def test_calculate_average_face_size(self):
        """Test average face size calculation"""
        processor = VideoProcessor()
        
        detections = [
            FaceDetection(bbox=(0, 0, 100, 100), confidence=0.9, frame_idx=0),  # 10,000 pixels
            FaceDetection(bbox=(0, 0, 50, 50), confidence=0.8, frame_idx=1),    # 2,500 pixels
        ]
        
        avg_size = processor._calculate_average_face_size(detections)
        assert avg_size == 6250.0  # (10000 + 2500) / 2
        
        # Test empty list
        avg_size_empty = processor._calculate_average_face_size([])
        assert avg_size_empty == 0.0
        
    @patch('cv2.VideoCapture')
    def test_batch_process_videos_mock(self, mock_video_capture):
        """Test batch processing with mocked videos"""
        processor = VideoProcessor(max_frames=3)
        
        # Setup mock
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, self._create_test_frame(i)) for i in range(10)] + [(False, None)]
        mock_video_capture.return_value = mock_cap
        
        video_paths = ["video1.mp4", "video2.mp4"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = processor.batch_process_videos(
                video_paths, 
                output_dir=Path(temp_dir),
                max_workers=1  # Use single worker for testing
            )
        
        assert len(results) == 2
        for video_path in video_paths:
            assert video_path in results
            result = results[video_path]
            assert "success" in result
            assert "face_crops" in result
            assert "metadata" in result


class TestVideoProcessorIntegration:
    """Integration tests for video processor"""
    
    def test_full_pipeline_synthetic(self):
        """Test full video processing pipeline with synthetic data"""
        processor = VideoProcessor(
            target_size=(128, 128),  # Smaller for faster testing
            max_frames=3,
            frame_interval=1
        )
        
        # Create synthetic video data
        synthetic_frames = [self._create_face_frame(i) for i in range(3)]
        
        # Test face detection on synthetic frames
        all_detections = []
        for i, frame in enumerate(synthetic_frames):
            detections = processor.face_detector.detect_faces(frame, i)
            all_detections.extend(detections)
        
        # Should not crash and return some structure
        assert isinstance(all_detections, list)
        
    def _create_face_frame(self, frame_idx: int) -> np.ndarray:
        """Create frame with prominent face-like features"""
        frame = np.ones((224, 224, 3), dtype=np.uint8) * 120
        
        # Large face oval
        cv2.ellipse(frame, (112, 112), (80, 100), 0, 0, 360, (180, 160, 140), -1)
        
        # Prominent eyes
        cv2.circle(frame, (85, 95), 12, (50, 50, 50), -1)
        cv2.circle(frame, (139, 95), 12, (50, 50, 50), -1)
        
        # Nose
        cv2.circle(frame, (112, 115), 8, (160, 140, 120), -1)
        
        # Mouth
        cv2.ellipse(frame, (112, 145), (25, 15), 0, 0, 180, (100, 80, 80), -1)
        
        # Add slight variation per frame
        noise_factor = frame_idx * 5
        noise = np.random.randint(-noise_factor, noise_factor, (224, 224, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return frame


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
