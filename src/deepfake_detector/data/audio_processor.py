"""
Audio Processing for Deepfake Detection

Handles audio loading, feature extraction, and preprocessing for deepfake detection.
Supports multiple audio formats and advanced feature extraction techniques.

Author: Your Name
"""

import librosa
import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioMetadata:
    """Container for audio file metadata"""
    filepath: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    size_mb: float
    format: str


@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    mfcc: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_bandwidth: np.ndarray
    zero_crossing_rate: np.ndarray
    mel_spectrogram: np.ndarray
    chroma: np.ndarray
    tonnetz: np.ndarray
    tempo: float
    spectral_contrast: np.ndarray


class AudioProcessor:
    """Main audio processing pipeline for deepfake detection"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 duration: float = 3.0,
                 hop_length: int = 512,
                 n_mfcc: int = 13,
                 n_mels: int = 128):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Target sample rate for resampling
            duration: Fixed duration for audio clips
            hop_length: Hop length for feature extraction
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of Mel frequency bands
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_samples = int(sample_rate * duration)
        
        logger.info(f"Initialized AudioProcessor: {sample_rate}Hz, {duration}s duration")
    
    def extract_audio_metadata(self, audio_path: Union[str, Path]) -> AudioMetadata:
        """Extract metadata from audio file"""
        audio_path = Path(audio_path)
        
        try:
            # Use soundfile for metadata (more reliable than librosa)
            info = sf.info(str(audio_path))
            
            return AudioMetadata(
                filepath=str(audio_path),
                duration=info.duration,
                sample_rate=info.samplerate,
                channels=info.channels,
                bit_depth=info.subtype_info.bits if info.subtype_info else 16,
                size_mb=audio_path.stat().st_size / (1024 * 1024),
                format=info.format
            )
        except Exception as e:
            logger.error(f"Error extracting metadata from {audio_path}: {e}")
            raise
    
    def load_audio(self, audio_path: Union[str, Path], 
                  offset: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file
        
        Args:
            audio_path: Path to audio file
            offset: Start time offset in seconds
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                duration=self.duration,
                offset=offset,
                mono=True  # Convert to mono
            )
            
            # Ensure fixed length
            if len(audio) > self.n_samples:
                audio = audio[:self.n_samples]
            else:
                # Pad with zeros if too short
                audio = np.pad(audio, (0, max(0, self.n_samples - len(audio))), 'constant')
            
            return audio, self.sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return silence if loading fails
            return np.zeros(self.n_samples), self.sample_rate
    
    def extract_features(self, audio: np.ndarray, 
                        sample_rate: Optional[int] = None) -> AudioFeatures:
        """
        Extract comprehensive audio features for deepfake detection
        
        Args:
            audio: Audio signal array
            sample_rate: Sample rate (uses default if None)
            
        Returns:
            AudioFeatures object with all extracted features
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        # MFCC features (most important for speech)
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=sample_rate, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length
        )
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sample_rate, hop_length=self.hop_length
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sample_rate, hop_length=self.hop_length
        )
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sample_rate, hop_length=self.hop_length
        )
        
        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            audio, hop_length=self.hop_length
        )
        
        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, 
            sr=sample_rate, 
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sample_rate, hop_length=self.hop_length
        )
        
        # Tonnetz (harmonic network) - handle low sample rates
        try:
            tonnetz = librosa.feature.tonnetz(
                y=librosa.effects.harmonic(audio), 
                sr=sample_rate
            )
        except Exception:
            # Fallback for low sample rates or other issues
            n_frames = 1 + int((len(audio) - 1) // self.hop_length)
            tonnetz = np.zeros((6, n_frames))
        
        # Tempo estimation
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        except:
            tempo = 0.0
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sample_rate, hop_length=self.hop_length
        )
        
        return AudioFeatures(
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_bandwidth=spectral_bandwidth,
            zero_crossing_rate=zero_crossing_rate,
            mel_spectrogram=mel_spectrogram,
            chroma=chroma,
            tonnetz=tonnetz,
            tempo=tempo,
            spectral_contrast=spectral_contrast
        )
    
    def create_augmentations(self, audio: np.ndarray, 
                           num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Create augmented versions of audio for training robustness
        
        Args:
            audio: Original audio signal
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented audio arrays
        """
        augmented_audios = [audio.copy()]  # Include original
        
        for i in range(num_augmentations):
            aug_audio = audio.copy()
            
            # Random combination of augmentations
            augmentation_choice = np.random.randint(0, 5)
            
            if augmentation_choice == 0:
                # Add white noise
                noise_factor = np.random.uniform(0.001, 0.01)
                noise = np.random.normal(0, noise_factor, len(aug_audio))
                aug_audio = aug_audio + noise
                
            elif augmentation_choice == 1:
                # Time stretch (speed change)
                try:
                    rate = np.random.uniform(0.8, 1.2)
                    aug_audio = librosa.effects.time_stretch(aug_audio, rate=rate)
                    # Ensure consistent length
                    if len(aug_audio) > len(audio):
                        aug_audio = aug_audio[:len(audio)]
                    else:
                        aug_audio = np.pad(aug_audio, (0, len(audio) - len(aug_audio)), 'constant')
                except:
                    pass  # Keep original if time stretch fails
                    
            elif augmentation_choice == 2:
                # Pitch shift
                try:
                    n_steps = np.random.uniform(-2, 2)
                    aug_audio = librosa.effects.pitch_shift(
                        aug_audio, sr=self.sample_rate, n_steps=n_steps
                    )
                except:
                    pass
                    
            elif augmentation_choice == 3:
                # Time shift
                shift_max = len(aug_audio) // 4
                shift = np.random.randint(-shift_max, shift_max)
                aug_audio = np.roll(aug_audio, shift)
                
            elif augmentation_choice == 4:
                # Volume change
                volume_factor = np.random.uniform(0.5, 1.5)
                aug_audio = aug_audio * volume_factor
                
            # Normalize to prevent clipping
            aug_audio = np.clip(aug_audio, -1.0, 1.0)
            augmented_audios.append(aug_audio)
        
        return augmented_audios
    
    def detect_voice_activity(self, audio: np.ndarray, 
                            frame_length: int = 2048,
                            hop_length: Optional[int] = None) -> np.ndarray:
        """
        Detect voice activity in audio signal
        
        Args:
            audio: Audio signal
            frame_length: Frame length for analysis
            hop_length: Hop length (uses default if None)
            
        Returns:
            Binary array indicating voice activity per frame
        """
        if hop_length is None:
            hop_length = self.hop_length
            
        # Compute short-time energy
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Compute zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, 
            frame_length=frame_length,
            hop_length=hop_length
        )[0]
        
        # Simple thresholding for voice activity detection
        energy_threshold = np.mean(energy) * 0.1
        zcr_threshold = np.mean(zcr) * 0.5
        
        # Voice activity: high energy AND moderate ZCR
        voice_activity = (energy > energy_threshold) & (zcr < zcr_threshold * 2)
        
        return voice_activity
    
    def extract_segments(self, audio_path: Union[str, Path], 
                        segment_length: float = 3.0,
                        overlap: float = 0.5) -> List[Tuple[np.ndarray, float]]:
        """
        Extract overlapping segments from long audio file
        
        Args:
            audio_path: Path to audio file
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments (0-1)
            
        Returns:
            List of (audio_segment, start_time) tuples
        """
        metadata = self.extract_audio_metadata(audio_path)
        segments = []
        
        step_size = segment_length * (1 - overlap)
        num_segments = int((metadata.duration - segment_length) / step_size) + 1
        
        for i in range(num_segments):
            start_time = i * step_size
            if start_time + segment_length <= metadata.duration:
                audio_segment, _ = self.load_audio(audio_path, offset=start_time)
                segments.append((audio_segment, start_time))
        
        return segments
    
    def analyze_audio_quality(self, audio_path: Union[str, Path]) -> Dict:
        """
        Analyze audio quality metrics relevant to deepfake detection
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with quality metrics
        """
        metadata = self.extract_audio_metadata(audio_path)
        audio, sr = self.load_audio(audio_path)
        features = self.extract_features(audio, sr)
        
        # Voice activity detection
        voice_activity = self.detect_voice_activity(audio)
        voice_ratio = np.sum(voice_activity) / len(voice_activity)
        
        # Signal-to-noise ratio estimation
        voice_frames = voice_activity
        if np.sum(voice_frames) > 0:
            voice_energy = np.mean(librosa.feature.rms(y=audio)[0][voice_frames])
            noise_energy = np.mean(librosa.feature.rms(y=audio)[0][~voice_frames])
            snr_estimate = 20 * np.log10(voice_energy / (noise_energy + 1e-8))
        else:
            snr_estimate = 0.0
        
        # Spectral quality metrics
        spectral_centroid_mean = np.mean(features.spectral_centroid)
        spectral_rolloff_mean = np.mean(features.spectral_rolloff)
        spectral_bandwidth_mean = np.mean(features.spectral_bandwidth)
        
        quality_metrics = {
            "sample_rate": metadata.sample_rate,
            "sample_rate_class": self._classify_sample_rate(metadata.sample_rate),
            "duration": metadata.duration,
            "channels": metadata.channels,
            "bit_depth": metadata.bit_depth,
            "voice_activity_ratio": voice_ratio,
            "snr_estimate_db": snr_estimate,
            "spectral_centroid_mean": spectral_centroid_mean,
            "spectral_rolloff_mean": spectral_rolloff_mean,
            "spectral_bandwidth_mean": spectral_bandwidth_mean,
            "tempo": features.tempo,
            "quality_score": 0.0  # Will be calculated
        }
        
        # Calculate overall quality score (0-1)
        score = 0.0
        
        # Sample rate contribution (25%)
        if quality_metrics["sample_rate_class"] == "high":
            score += 0.25
        elif quality_metrics["sample_rate_class"] == "standard":
            score += 0.20
        elif quality_metrics["sample_rate_class"] == "telephone":
            score += 0.10
        
        # Voice activity ratio contribution (25%)
        score += min(voice_ratio * 1.5, 1.0) * 0.25
        
        # SNR contribution (25%)
        snr_normalized = max(0, min(snr_estimate / 30.0, 1.0))
        score += snr_normalized * 0.25
        
        # Duration contribution (25%)
        if 2.0 <= quality_metrics["duration"] <= 10.0:
            score += 0.25
        elif quality_metrics["duration"] >= 1.0:
            score += 0.15
        
        quality_metrics["quality_score"] = score
        
        return quality_metrics
    
    def _classify_sample_rate(self, sample_rate: int) -> str:
        """Classify audio sample rate"""
        if sample_rate >= 44100:
            return "high"
        elif sample_rate >= 16000:
            return "standard"
        elif sample_rate >= 8000:
            return "telephone"
        else:
            return "low"
    
    def batch_process_audio(self, audio_paths: List[Union[str, Path]],
                           output_dir: Optional[Path] = None,
                           extract_features: bool = True,
                           max_workers: int = 4) -> Dict[str, Dict]:
        """
        Process multiple audio files in parallel
        
        Args:
            audio_paths: List of audio file paths
            output_dir: Directory to save processed features
            extract_features: Whether to extract features
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary with processing results for each file
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_path = {}
            for path in audio_paths:
                if extract_features:
                    future = executor.submit(self._process_single_audio_with_features, path)
                else:
                    future = executor.submit(self._process_single_audio, path)
                future_to_path[future] = path
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_path),
                             total=len(audio_paths),
                             desc="Processing audio files"):
                audio_path = future_to_path[future]
                try:
                    result = future.result()
                    
                    # Save features if output directory specified
                    if output_dir and extract_features:
                        output_dir = Path(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        audio_name = Path(audio_path).stem
                        feature_path = output_dir / f"{audio_name}_features.npz"
                        
                        # Save features as compressed numpy arrays
                        feature_dict = {}
                        if 'features' in result:
                            features = result['features']
                            feature_dict = {
                                'mfcc': features.mfcc,
                                'spectral_centroid': features.spectral_centroid,
                                'spectral_rolloff': features.spectral_rolloff,
                                'spectral_bandwidth': features.spectral_bandwidth,
                                'zero_crossing_rate': features.zero_crossing_rate,
                                'mel_spectrogram': features.mel_spectrogram,
                                'chroma': features.chroma,
                                'tonnetz': features.tonnetz,
                                'spectral_contrast': features.spectral_contrast,
                                'tempo': np.array([features.tempo])
                            }
                        
                        np.savez_compressed(feature_path, **feature_dict)
                        
                        # Save metadata
                        metadata_path = output_dir / f"{audio_name}_metadata.json"
                        with open(metadata_path, 'w') as f:
                            serializable_metadata = self._make_serializable(result['metadata'])
                            json.dump(serializable_metadata, f, indent=2)
                    
                    results[str(audio_path)] = {
                        **result,
                        "success": True
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing {audio_path}: {e}")
                    results[str(audio_path)] = {
                        "audio": None,
                        "features": None,
                        "metadata": {},
                        "success": False,
                        "error": str(e)
                    }
        
        return results
    
    def _process_single_audio(self, audio_path: Union[str, Path]) -> Dict:
        """Process single audio file without feature extraction"""
        audio, sr = self.load_audio(audio_path)
        metadata = self.extract_audio_metadata(audio_path)
        
        return {
            "audio": audio,
            "metadata": metadata
        }
    
    def _process_single_audio_with_features(self, audio_path: Union[str, Path]) -> Dict:
        """Process single audio file with feature extraction"""
        audio, sr = self.load_audio(audio_path)
        features = self.extract_features(audio, sr)
        metadata = self.extract_audio_metadata(audio_path)
        quality_metrics = self.analyze_audio_quality(audio_path)
        
        return {
            "audio": audio,
            "features": features,
            "metadata": metadata,
            "quality_metrics": quality_metrics
        }
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, AudioMetadata):
            return {
                "filepath": obj.filepath,
                "duration": obj.duration,
                "sample_rate": obj.sample_rate,
                "channels": obj.channels,
                "bit_depth": obj.bit_depth,
                "size_mb": obj.size_mb,
                "format": obj.format
            }
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        else:
            return obj
    
    def visualize_features(self, audio_path: Union[str, Path], 
                          save_path: Optional[Path] = None) -> None:
        """
        Create visualization of audio features
        
        Args:
            audio_path: Path to audio file
            save_path: Path to save visualization (optional)
        """
        audio, sr = self.load_audio(audio_path)
        features = self.extract_features(audio, sr)
        
        # Create subplot layout
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Audio Features: {Path(audio_path).name}', fontsize=16)
        
        # Time axis for features
        time_frames = np.linspace(0, self.duration, features.mfcc.shape[1])
        
        # MFCC
        axes[0, 0].imshow(features.mfcc, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('MFCC')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('MFCC Coefficient')
        
        # Mel Spectrogram
        mel_db = librosa.power_to_db(features.mel_spectrogram, ref=np.max)
        axes[0, 1].imshow(mel_db, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Mel Spectrogram')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Mel Frequency')
        
        # Spectral Centroid
        axes[1, 0].plot(time_frames, features.spectral_centroid[0])
        axes[1, 0].set_title('Spectral Centroid')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Hz')
        
        # Zero Crossing Rate
        axes[1, 1].plot(time_frames, features.zero_crossing_rate[0])
        axes[1, 1].set_title('Zero Crossing Rate')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Rate')
        
        # Chroma
        axes[2, 0].imshow(features.chroma, aspect='auto', origin='lower', cmap='viridis')
        axes[2, 0].set_title('Chroma')
        axes[2, 0].set_xlabel('Time')
        axes[2, 0].set_ylabel('Chroma')
        
        # Spectral Contrast
        axes[2, 1].imshow(features.spectral_contrast, aspect='auto', origin='lower', cmap='viridis')
        axes[2, 1].set_title('Spectral Contrast')
        axes[2, 1].set_xlabel('Time')
        axes[2, 1].set_ylabel('Frequency Band')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature visualization to {save_path}")
        
        plt.show()


def main():
    """Demonstration of audio processing functionality"""
    # Initialize processor
    processor = AudioProcessor(
        sample_rate=16000,
        duration=3.0,
        n_mfcc=13,
        n_mels=128
    )
    
    print("🎵 Audio Processing Pipeline Demo")
    print("=" * 50)
    
    print("\nCapabilities:")
    print("• Multi-format audio loading (WAV, MP3, FLAC, etc.)")
    print("• Feature extraction (MFCC, spectral, chroma, etc.)")
    print("• Voice activity detection")
    print("• Audio quality analysis")
    print("• Data augmentation (noise, pitch shift, time stretch)")
    print("• Batch processing with parallel execution")
    print("• Segment extraction from long files")
    print("• Feature visualization")
    
    print(f"\nProcessor Configuration:")
    print(f"• Sample rate: {processor.sample_rate} Hz")
    print(f"• Duration: {processor.duration} seconds")
    print(f"• MFCC coefficients: {processor.n_mfcc}")
    print(f"• Mel bands: {processor.n_mels}")
    print(f"• Hop length: {processor.hop_length}")
    
    # Example usage
    print(f"\nExample Usage:")
    print(f"```python")
    print(f"# Load and extract features")
    print(f"audio, sr = processor.load_audio('audio.wav')")
    print(f"features = processor.extract_features(audio)")
    print(f"")
    print(f"# Create augmentations")
    print(f"augmented = processor.create_augmentations(audio)")
    print(f"")
    print(f"# Batch process files")
    print(f"results = processor.batch_process_audio(['file1.wav', 'file2.wav'])")
    print(f"")
    print(f"# Analyze quality")
    print(f"quality = processor.analyze_audio_quality('audio.wav')")
    print(f"```")
    
    print(f"\nFeature Dimensions:")
    print(f"• MFCC: ({processor.n_mfcc}, time_frames)")
    print(f"• Mel Spectrogram: ({processor.n_mels}, time_frames)")
    print(f"• Spectral features: (1, time_frames)")
    print(f"• Chroma: (12, time_frames)")
    print(f"• Tonnetz: (6, time_frames)")


if __name__ == "__main__":
    main()
