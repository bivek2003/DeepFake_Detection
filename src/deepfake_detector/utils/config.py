"""
Configuration and Utilities for Deepfake Detection

Handles configuration management, logging setup, and utility functions.

Author: Bivek Sharma Panthi
"""

import yaml
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List  # Added List import
from dataclasses import dataclass, asdict
import torch
import numpy as np
from datetime import datetime
import sys


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Paths
    data_root: str = "./datasets"
    processed_root: str = "./preprocessed"
    
    # Video settings
    video_target_size: List = [224, 224]  # Changed from tuple to list for YAML compatibility
    max_frames_per_video: int = 30
    frame_interval: int = 1
    
    # Audio settings  
    audio_sample_rate: int = 16000
    audio_duration: float = 3.0
    n_mfcc: int = 13
    n_mels: int = 128
    hop_length: int = 512
    
    # Splitting
    test_size: float = 0.2
    val_size: float = 0.1
    stratify: bool = True
    random_state: int = 42


@dataclass  
class TrainingConfig:
    """Training configuration (for Phase 2)"""
    # Model settings
    model_name: str = "efficientnet_b4"
    num_classes: int = 2
    pretrained: bool = True
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 50
    weight_decay: float = 1e-5
    
    # Optimization
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    save_every: int = 5
    save_best: bool = True
    early_stopping_patience: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    log_dir: str = "./logs"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig
    training: TrainingConfig
    logging: LoggingConfig
    
    # Metadata
    version: str = "0.1.0"
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ConfigManager:
    """Configuration management system"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = config_path
        self.config = None
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            self.config = self.create_default_config()
    
    def create_default_config(self) -> Config:
        """Create default configuration"""
        return Config(
            data=DataConfig(),
            training=TrainingConfig(),
            logging=LoggingConfig()
        )
    
    def load_config(self, config_path: Union[str, Path]) -> Config:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert dict to Config object
        self.config = Config(
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            version=config_dict.get('version', '0.1.0'),
            created_at=config_dict.get('created_at', datetime.now().isoformat())
        )
        
        return self.config
    
    def save_config(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        if self.config is None:
            raise ValueError("No configuration to save")
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(self.config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logging.info(f"Configuration saved to {config_path}")
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values"""
        if self.config is None:
            self.config = self.create_default_config()
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                if isinstance(getattr(self.config, key), dict):
                    getattr(self.config, key).update(value)
                else:
                    setattr(self.config, key, value)
            else:
                logging.warning(f"Unknown configuration key: {key}")
    
    def get_config(self) -> Config:
        """Get current configuration"""
        if self.config is None:
            self.config = self.create_default_config()
        return self.config


class LoggingSetup:
    """Centralized logging setup"""
    
    @staticmethod
    def setup_logging(config: LoggingConfig) -> logging.Logger:
        """Setup logging configuration"""
        # Create logs directory
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, config.level.upper()))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(config.format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if config.file_logging:
            from logging.handlers import RotatingFileHandler
            
            log_file = log_dir / "deepfake_detector.log"
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=config.max_bytes,
                backupCount=config.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


class DeviceManager:
    """Device management for PyTorch"""
    
    @staticmethod
    def get_device(device_preference: str = "auto") -> torch.device:
        """Get the appropriate device for computation"""
        if device_preference == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logging.info("Using CPU")
        elif device_preference == "cuda":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                logging.warning("CUDA not available, falling back to CPU")
                device = torch.device("cpu")
        else:
            device = torch.device(device_preference)
            logging.info(f"Using device: {device}")
        
        return device
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get detailed device information"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
        
        if torch.cuda.is_available():
            info["devices"] = []
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "id": i,
                    "name": device_props.name,
                    "total_memory": device_props.total_memory,
                    "multiprocessor_count": device_props.multi_processor_count,
                })
        
        return info


class MetricsCalculator:
    """Utility class for calculating various metrics"""
    
    @staticmethod
    def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate accuracy"""
        return np.mean(predictions == targets)
    
    @staticmethod
    def calculate_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(targets, predictions)
    
    @staticmethod
    def calculate_classification_report(predictions: np.ndarray, 
                                      targets: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed classification metrics"""
        from sklearn.metrics import classification_report, precision_recall_fscore_support
        
        # Basic metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, predictions, average='macro'
        )
        
        return {
            "accuracy": MetricsCalculator.calculate_accuracy(predictions, targets),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "class_names": ["Real", "Fake"]
        }


class FileUtils:
    """File handling utilities"""
    
    @staticmethod
    def ensure_dir(directory: Union[str, Path]) -> Path:
        """Ensure directory exists"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    
    @staticmethod
    def get_file_size_mb(filepath: Union[str, Path]) -> float:
        """Get file size in MB"""
        return Path(filepath).stat().st_size / (1024 * 1024)
    
    @staticmethod
    def list_files(directory: Union[str, Path], 
                  pattern: str = "*",
                  recursive: bool = True) -> List[Path]:
        """List files in directory matching pattern"""
        directory = Path(directory)
        if not directory.exists():
            return []
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    
    @staticmethod
    def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save data to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load data from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_video_extensions() -> List[str]:
        """Get list of supported video extensions"""
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    @staticmethod
    def get_audio_extensions() -> List[str]:
        """Get list of supported audio extensions"""
        return ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.wma', '.aac']
    
    @staticmethod
    def filter_media_files(filepaths: List[Path], 
                          media_type: str = "both") -> List[Path]:
        """Filter files by media type"""
        if media_type == "video":
            extensions = FileUtils.get_video_extensions()
        elif media_type == "audio":
            extensions = FileUtils.get_audio_extensions()
        elif media_type == "both":
            extensions = FileUtils.get_video_extensions() + FileUtils.get_audio_extensions()
        else:
            raise ValueError("media_type must be 'video', 'audio', or 'both'")
        
        return [f for f in filepaths if f.suffix.lower() in extensions]


class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_file_exists(filepath: Union[str, Path]) -> bool:
        """Validate that file exists"""
        return Path(filepath).exists()
    
    @staticmethod
    def validate_directory_structure(data_root: Union[str, Path], 
                                   required_subdirs: List[str]) -> Dict[str, bool]:
        """Validate directory structure"""
        data_root = Path(data_root)
        results = {}
        
        for subdir in required_subdirs:
            subdir_path = data_root / subdir
            results[subdir] = subdir_path.exists() and subdir_path.is_dir()
        
        return results
    
    @staticmethod
    def validate_config(config: Config) -> List[str]:
        """Validate configuration settings"""
        errors = []
        
        # Data validation
        if config.data.test_size + config.data.val_size >= 1.0:
            errors.append("test_size + val_size must be < 1.0")
        
        if config.data.audio_sample_rate <= 0:
            errors.append("audio_sample_rate must be positive")
        
        if config.data.audio_duration <= 0:
            errors.append("audio_duration must be positive")
        
        # Training validation
        if config.training.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if config.training.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if config.training.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        
        return errors
    
    @staticmethod
    def validate_data_splits(train_paths: List[str], 
                           val_paths: List[str], 
                           test_paths: List[str]) -> Dict[str, Any]:
        """Validate data splits for overlaps and completeness"""
        train_set = set(train_paths)
        val_set = set(val_paths)
        test_set = set(test_paths)
        
        return {
            "train_val_overlap": len(train_set & val_set),
            "train_test_overlap": len(train_set & test_set),
            "val_test_overlap": len(val_set & test_set),
            "total_unique": len(train_set | val_set | test_set),
            "total_files": len(train_paths) + len(val_paths) + len(test_paths),
            "has_overlaps": bool(train_set & val_set) or bool(train_set & test_set) or bool(val_set & test_set)
        }


class ProgressTracker:
    """Progress tracking utility"""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
        
    def update(self, step: int = 1, message: Optional[str] = None):
        """Update progress"""
        self.current_step += step
        progress = self.current_step / self.total_steps
        
        # Calculate ETA
        elapsed = datetime.now() - self.start_time
        if progress > 0:
            eta = elapsed / progress * (1 - progress)
        else:
            eta = None
        
        # Format message
        status = f"{self.description}: {self.current_step}/{self.total_steps} ({progress:.1%})"
        if eta:
            status += f" - ETA: {eta}"
        if message:
            status += f" - {message}"
        
        logging.info(status)
    
    def finish(self, message: Optional[str] = None):
        """Mark as finished"""
        elapsed = datetime.now() - self.start_time
        status = f"{self.description} completed in {elapsed}"
        if message:
            status += f" - {message}"
        logging.info(status)


class SystemUtils:
    """System utilities"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get system information"""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_usage": {
                "total_gb": psutil.disk_usage('.').total / (1024**3),
                "used_gb": psutil.disk_usage('.').used / (1024**3),
                "free_gb": psutil.disk_usage('.').free / (1024**3)
            },
            "torch_version": torch.__version__,
            "device_info": DeviceManager.get_device_info()
        }
    
    @staticmethod
    def check_dependencies() -> Dict[str, str]:
        """Check if all dependencies are installed"""
        dependencies = {
            "torch": "PyTorch",
            "torchvision": "TorchVision", 
            "opencv-cv2": "OpenCV",
            "librosa": "Librosa",
            "sklearn": "Scikit-learn",
            "numpy": "NumPy",
            "pandas": "Pandas",
            "matplotlib": "Matplotlib",
            "tqdm": "TQDM",
            "yaml": "PyYAML"
        }
        
        status = {}
        for module, name in dependencies.items():
            try:
                if module == "opencv-cv2":
                    import cv2
                    status[name] = f"✅ {cv2.__version__}"
                else:
                    mod = __import__(module)
                    version = getattr(mod, '__version__', 'Unknown')
                    status[name] = f"✅ {version}"
            except ImportError:
                status[name] = "❌ Not installed"
        
        return status


# Create default configuration file
def create_default_config_file(config_path: str = "config.yaml"):
    """Create default configuration file"""
    config_manager = ConfigManager()
    config = config_manager.create_default_config()
    
    config_dict = asdict(config)
    
    # Add comments to the YAML
    yaml_content = f"""# Deepfake Detection Configuration
# Generated on {datetime.now().isoformat()}

version: "{config.version}"
created_at: "{config.created_at}"

# Data processing settings
data:
  data_root: "{config.data.data_root}"
  processed_root: "{config.data.processed_root}"
  
  # Video processing
  video_target_size: [{config.data.video_target_size[0]}, {config.data.video_target_size[1]}]
  max_frames_per_video: {config.data.max_frames_per_video}
  frame_interval: {config.data.frame_interval}
  
  # Audio processing  
  audio_sample_rate: {config.data.audio_sample_rate}
  audio_duration: {config.data.audio_duration}
  n_mfcc: {config.data.n_mfcc}
  n_mels: {config.data.n_mels}
  hop_length: {config.data.hop_length}
  
  # Data splitting
  test_size: {config.data.test_size}
  val_size: {config.data.val_size}
  stratify: {config.data.stratify}
  random_state: {config.data.random_state}

# Training settings (Phase 2)
training:
  # Model
  model_name: "{config.training.model_name}"
  num_classes: {config.training.num_classes}
  pretrained: {config.training.pretrained}
  
  # Hyperparameters
  batch_size: {config.training.batch_size}
  learning_rate: {config.training.learning_rate}
  num_epochs: {config.training.num_epochs}
  weight_decay: {config.training.weight_decay}
  
  # Optimization
  optimizer: "{config.training.optimizer}"
  scheduler: "{config.training.scheduler}"
  warmup_epochs: {config.training.warmup_epochs}
  
  # Hardware
  device: "{config.training.device}"
  num_workers: {config.training.num_workers}
  pin_memory: {config.training.pin_memory}
  
  # Checkpointing
  save_every: {config.training.save_every}
  save_best: {config.training.save_best}
  early_stopping_patience: {config.training.early_stopping_patience}

# Logging settings
logging:
  level: "{config.logging.level}"
  format: "{config.logging.format}"
  file_logging: {config.logging.file_logging}
  log_dir: "{config.logging.log_dir}"
  max_bytes: {config.logging.max_bytes}
  backup_count: {config.logging.backup_count}
"""
    
    with open(config_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created default configuration file: {config_path}")
    return config_path


def main():
    """Demonstration of configuration and utilities"""
    print("⚙️  Configuration & Utilities Demo")
    print("=" * 50)
    
    # Create config manager
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Setup logging
    logger = LoggingSetup.setup_logging(config.logging)
    logger.info("Logging system initialized")
    
    # Check system info
    system_info = SystemUtils.get_system_info()
    print(f"\n💻 System Information:")
    print(f"• Platform: {system_info['platform']}")
    print(f"• Python: {system_info['python_version']}")
    print(f"• CPU cores: {system_info['cpu_count']}")
    print(f"• Memory: {system_info['memory_total_gb']:.1f} GB")
    print(f"• PyTorch: {system_info['torch_version']}")
    
    # Check dependencies
    deps = SystemUtils.check_dependencies()
    print(f"\n📦 Dependencies:")
    for name, status in deps.items():
        print(f"• {name}: {status}")
    
    # Device info
    device = DeviceManager.get_device()
    device_info = DeviceManager.get_device_info()
    print(f"\n🔧 Device Information:")
    print(f"• Current device: {device}")
    print(f"• CUDA available: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"• GPU count: {device_info['device_count']}")
        for gpu in device_info['devices']:
            print(f"  - {gpu['name']}: {gpu['total_memory'] / 1e9:.1f} GB")
    
    # Validate config
    validation_errors = ValidationUtils.validate_config(config)
    print(f"\n✅ Configuration Validation:")
    if validation_errors:
        print("❌ Errors found:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid")
    
    # Create default config file
    config_path = create_default_config_file("config_example.yaml")
    
    print(f"\n🎯 Features:")
    print("• YAML configuration management")
    print("• Centralized logging setup")
    print("• Device auto-detection (CPU/GPU)")
    print("• System information gathering")
    print("• File handling utilities")
    print("• Data validation tools")
    print("• Progress tracking")
    print("• Metrics calculation")


if __name__ == "__main__":
    main()
