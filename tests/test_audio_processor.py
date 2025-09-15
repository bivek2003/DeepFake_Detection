"""
Unit tests for AudioProcessor

Tests feature extraction, audio augmentations, VAD, and batch processing.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_detector.data.audio_processor import AudioProcessor, AudioFeatures, AudioMetadata

@pytest.fixture
def audio_processor():
    """Fixture for AudioProcessor"""
    return AudioProcessor(sample_rate=16000, duration=3.0, n_mfcc=13, n_mels=128)

class TestAudioProcessorUnit:
    """Unit tests for AudioProcessor"""

    def _create_synthetic_audio(self, processor: AudioProcessor, freq: float = 440.0) -> np.ndarray:
        """Generate a simple sine wave as synthetic audio"""
        t = np.linspace(0, processor.duration, processor.n_samples, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        return audio

    def test_initialization(self, audio_processor):
        """Test processor initialization"""
        assert audio_processor.sample_rate == 16000
        assert audio_processor.n_samples == 16000 * 3
        assert audio_processor.n_mfcc == 13

    def test_extract_features(self, audio_processor):
        """Test feature extraction on synthetic audio"""
        audio = self._create_synthetic_audio(audio_processor)
        features: AudioFeatures = audio_processor.extract_features(audio)
        assert isinstance(features, AudioFeatures)
        assert features.mfcc.shape[0] == audio_processor.n_mfcc
        assert features.mel_spectrogram.shape[0] == audio_processor.n_mels
        assert isinstance(features.tempo, float)

    def test_detect_voice_activity(self, audio_processor):
        """Test voice activity detection"""
        audio = self._create_synthetic_audio(audio_processor)
        vad_mask = audio_processor.detect_voice_activity(audio)
        assert isinstance(vad_mask, np.ndarray)
        assert vad_mask.shape[0] > 0
        assert vad_mask.dtype == bool or vad_mask.dtype == np.bool_

    def test_create_augmentations(self, audio_processor):
        """Test audio augmentation"""
        audio = self._create_synthetic_audio(audio_processor)
        augmented = audio_processor.create_augmentations(audio, num_augmentations=3)
        assert isinstance(augmented, list)
        assert len(augmented) == 4  # original + 3 augmentations
        for aug in augmented:
            assert aug.shape == audio.shape
            assert np.max(np.abs(aug)) <= 1.0  # Ensure clipping is applied

    def test_extract_audio_metadata_mock(self, audio_processor):
        """Test audio metadata extraction with mock"""
        with patch("soundfile.info") as mock_info:
            mock_info.return_value.duration = 3.0
            mock_info.return_value.samplerate = 16000
            mock_info.return_value.channels = 1
            mock_info.return_value.format = "WAV"
            mock_info.return_value.subtype_info = None

            # Mock Path stat
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1024 * 1024 * 2  # 2 MB

                metadata: AudioMetadata = audio_processor.extract_audio_metadata("dummy.wav")
                assert isinstance(metadata, AudioMetadata)
                assert metadata.duration == 3.0
                assert metadata.sample_rate == 16000
                assert metadata.channels == 1
                assert metadata.size_mb == 2.0
                assert metadata.format == "WAV"

    @patch.object(AudioProcessor, "extract_features")
    @patch.object(AudioProcessor, "extract_audio_metadata")
    @patch.object(AudioProcessor, "load_audio")
    def test_batch_process_audio_mock(self, mock_load, mock_metadata, mock_features, audio_processor):
        """Test batch processing with mocked methods"""
        audio_array = np.zeros(audio_processor.n_samples)
        mock_load.return_value = (audio_array, audio_processor.sample_rate)
        mock_metadata.return_value = AudioMetadata(
            filepath="dummy.wav", duration=3.0, sample_rate=16000, channels=1, bit_depth=16, size_mb=1.0, format="WAV"
        )
        mock_features.return_value = AudioFeatures(
            mfcc=np.zeros((audio_processor.n_mfcc, 10)),
            spectral_centroid=np.zeros((1,10)),
            spectral_rolloff=np.zeros((1,10)),
            spectral_bandwidth=np.zeros((1,10)),
            zero_crossing_rate=np.zeros((1,10)),
            mel_spectrogram=np.zeros((audio_processor.n_mels,10)),
            chroma=np.zeros((12,10)),
            tonnetz=np.zeros((6,10)),
            tempo=120.0,
            spectral_contrast=np.zeros((7,10))
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            results = audio_processor.batch_process_audio(["audio1.wav", "audio2.wav"], output_dir=Path(tmpdir), max_workers=1)
        
        assert len(results) == 2
        for key, result in results.items():
            assert result["success"] is True
            assert "features" in result
            assert "metadata" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

