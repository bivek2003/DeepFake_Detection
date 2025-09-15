import pytest
import numpy as np
from deepfake_detector.data.audio_processor import AudioProcessor, AudioFeatures

@pytest.fixture
def audio_processor():
    return AudioProcessor(sample_rate=16000, duration=3.0)

def _get_synthetic_audio(processor: AudioProcessor):
    """Generate synthetic audio for testing"""
    # Simple sine wave
    t = np.linspace(0, processor.duration, processor.n_samples, endpoint=False)
    freq = 440.0  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio

def test_initialization(audio_processor):
    """Test processor initialization"""
    assert audio_processor.sample_rate == 16000
    assert audio_processor.n_samples == int(16000 * 3.0)
    assert audio_processor.n_mfcc == 13

def test_feature_extraction(audio_processor):
    """Verify MFCC, Mel-spectrogram, tempo extraction"""
    audio = _get_synthetic_audio(audio_processor)
    features: AudioFeatures = audio_processor.extract_features(audio)
    
    assert isinstance(features, AudioFeatures)
    assert features.mfcc.shape[0] == audio_processor.n_mfcc
    assert features.mel_spectrogram.shape[0] == audio_processor.n_mels
    assert isinstance(features.tempo, float)

def test_detect_voice_activity(audio_processor):
    """Test voice activity detection"""
    audio = _get_synthetic_audio(audio_processor)
    vad_mask = audio_processor.detect_voice_activity(audio)
    assert isinstance(vad_mask, np.ndarray)
    assert vad_mask.shape[0] > 0
    assert vad_mask.dtype == bool or vad_mask.dtype == np.bool_

def test_augmentations(audio_processor):
    """Test audio augmentation: time stretch, pitch shift, noise, etc."""
    audio = _get_synthetic_audio(audio_processor)
    augmented_list = audio_processor.create_augmentations(audio, num_augmentations=3)
    assert isinstance(augmented_list, list)
    assert len(augmented_list) == 4  # original + 3 augmentations
    for aug in augmented_list:
        assert aug.shape == audio.shape

