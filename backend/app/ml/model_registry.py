"""
Model registry for managing detection models.
Supports both demo mode and production mode with trained weights.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, List
import os

import torch
import torch.nn as nn

from app.logging_config import get_logger
from app.settings import get_settings

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Model information container."""
    name: str
    version: str
    device: str
    is_demo: bool
    backbone: str = "efficientnet_b4"
    calibration_method: str | None = None
    commit_hash: str | None = None
    metrics: dict[str, Any] | None = None
    threshold: float = 0.5
    temperature: float = 1.0


class DemoModel(nn.Module):
    """
    Demo model that produces deterministic but plausible outputs.
    Used when no real weights are available.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "demo-detector"
        self.version = "1.0.0-demo"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate deterministic fake scores based on image statistics.
        Creates plausible variation without real detection.
        
        Returns single logit per sample for binary classification.
        """
        batch_size = x.shape[0]
        
        # Use image statistics to generate pseudo-random but deterministic scores
        mean_values = x.mean(dim=(2, 3))  # [B, C]
        std_values = x.std(dim=(2, 3))    # [B, C]
        
        # Combine statistics into a score
        score_base = (mean_values.mean(dim=1) + std_values.mean(dim=1)) / 2
        
        # Add slight variation using image content
        noise = torch.sin(x.sum(dim=(1, 2, 3)) * 0.001) * 0.1
        logit = score_base - 0.5 + noise  # Centered around 0
        
        # Return as [B, 1] for binary classification
        return logit.unsqueeze(1)


class ModelRegistry:
    """
    Registry for managing detection models.
    Handles loading, caching, and inference with CUDA support.
    """
    
    # Supported model files in order of preference
    WEIGHT_FILES = [
        "deepfake_detector.pt",
        "best_model.pt", 
        "checkpoint.pt",
        "detector.pth",
        "model.pt",
    ]
    
    def __init__(self):
        self.model: nn.Module | None = None
        self.model_info: ModelInfo | None = None
        self.is_initialized = False
        self._device: str = "cpu"
        self._temperature: float = 1.0
        self._threshold: float = 0.5
        
        settings = get_settings()
        self._demo_mode = settings.demo_mode
        self._weights_path = Path(settings.model_weights_path)
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.model_info.name if self.model_info else "unknown"
    
    @property
    def model_version(self) -> str:
        """Get model version."""
        return self.model_info.version if self.model_info else "unknown"
    
    @property
    def device(self) -> str:
        """Get device being used."""
        return self._device
    
    @property
    def commit_hash(self) -> str | None:
        """Get commit hash if available."""
        return self.model_info.commit_hash if self.model_info else None
    
    @property
    def calibration_method(self) -> str | None:
        """Get calibration method."""
        return self.model_info.calibration_method if self.model_info else None
    
    @property
    def metrics(self) -> dict[str, Any] | None:
        """Get performance metrics."""
        return self.model_info.metrics if self.model_info else None
    
    @property
    def is_demo(self) -> bool:
        """Check if running in demo mode."""
        return self.model_info.is_demo if self.model_info else True
    
    def _detect_device(self) -> str:
        """Detect best available device."""
        settings = get_settings()
        
        if settings.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        else:
            device = settings.device
        
        return device
    
    def _find_weights_file(self) -> Optional[Path]:
        """Find model weights file."""
        if not self._weights_path.exists():
            logger.warning(f"Weights directory does not exist: {self._weights_path}")
            return None
        
        for filename in self.WEIGHT_FILES:
            weights_file = self._weights_path / filename
            if weights_file.exists():
                logger.info(f"Found weights file: {weights_file}")
                return weights_file
        
        # Check for any .pt or .pth file
        pt_files = list(self._weights_path.glob("*.pt")) + list(self._weights_path.glob("*.pth"))
        if pt_files:
            logger.info(f"Found weights file: {pt_files[0]}")
            return pt_files[0]
        
        logger.warning(f"No weights file found in {self._weights_path}")
        return None
    
    async def initialize(self) -> None:
        """Initialize model registry."""
        # Determine device
        self._device = self._detect_device()
        
        # Check for real weights
        weights_file = self._find_weights_file()
        
        if not self._demo_mode and weights_file is not None:
            await self._load_real_model(weights_file)
        else:
            if not self._demo_mode:
                logger.warning("Demo mode disabled but no weights found, using demo model")
            await self._load_demo_model()
        
        self.is_initialized = True
        logger.info(
            f"Model initialized",
            extra={
                "model_name": self.model_name,
                "model_version": self.model_version,
                "demo_mode": self.is_demo,
                "device": self._device,
            },
        )
    
    async def _load_demo_model(self) -> None:
        """Load demo model for development/testing."""
        self.model = DemoModel()
        self.model.to(self._device)
        self.model.eval()
        
        self.model_info = ModelInfo(
            name="demo-detector",
            version="1.0.0-demo",
            device=self._device,
            is_demo=True,
            calibration_method="none",
            metrics={
                "note": "Demo mode - scores are deterministic but not real predictions"
            },
        )
        
        logger.info("Demo model loaded")
    
    async def _load_real_model(self, weights_path: Path) -> None:
        """Load real model from weights file."""
        try:
            # Import real model architecture
            from app.ml.architecture import DeepfakeDetector
            
            logger.info(f"Loading model from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self._device)
            
            # Determine backbone from checkpoint
            if isinstance(checkpoint, dict):
                backbone = checkpoint.get("backbone", "efficientnet_b4")
                if "metadata" in checkpoint:
                    backbone = checkpoint["metadata"].get("backbone", backbone)
                if "config" in checkpoint:
                    backbone = checkpoint["config"].get("model", {}).get("backbone", backbone)
            else:
                backbone = "efficientnet_b4"
            
            logger.info(f"Creating model with backbone: {backbone}")
            
            # Create model
            self.model = DeepfakeDetector(
                backbone=backbone,
                pretrained=False,  # We're loading our own weights
            )
            
            # Load state dict
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    # Try to load directly if checkpoint is just a state dict
                    self.model.load_state_dict(checkpoint)
                
                # Load calibration parameters
                self._temperature = checkpoint.get("temperature", 1.0)
                self._threshold = checkpoint.get("threshold", 0.5)
                
                # Get metrics
                metrics = checkpoint.get("metrics", {})
                if "metadata" in checkpoint:
                    metrics.update(checkpoint["metadata"].get("metrics", {}))
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self._device)
            self.model.eval()
            
            # Enable inference optimizations
            if self._device == "cuda":
                torch.backends.cudnn.benchmark = True
            
            self.model_info = ModelInfo(
                name=checkpoint.get("model_name", "deepfake-detector") if isinstance(checkpoint, dict) else "deepfake-detector",
                version=checkpoint.get("version", "1.0.0") if isinstance(checkpoint, dict) else "1.0.0",
                device=self._device,
                is_demo=False,
                backbone=backbone,
                calibration_method="temperature_scaling" if self._temperature != 1.0 else None,
                commit_hash=checkpoint.get("commit_hash") if isinstance(checkpoint, dict) else None,
                metrics=metrics if isinstance(checkpoint, dict) else None,
                threshold=self._threshold,
                temperature=self._temperature,
            )
            
            logger.info(
                f"Real model loaded",
                extra={
                    "backbone": backbone,
                    "temperature": self._temperature,
                    "threshold": self._threshold,
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load real model: {e}")
            logger.info("Falling back to demo model")
            await self._load_demo_model()
    
    def predict(self, input_tensor: torch.Tensor) -> Tuple[float, float]:
        """
        Run prediction on input tensor.
        
        Args:
            input_tensor: Preprocessed image tensor [B, C, H, W]
            
        Returns:
            Tuple of (fake_probability, confidence)
            - fake_probability: Probability that image is fake (0-1)
            - confidence: Same as fake_probability for binary classification
        """
        if not self.is_initialized or self.model is None:
            raise RuntimeError("Model not initialized")
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self._device)
            
            # Get logits
            logits = self.model(input_tensor)
            
            # Apply temperature scaling
            if self._temperature != 1.0:
                logits = logits / self._temperature
            
            # Binary classification: single logit → sigmoid
            probs = torch.sigmoid(logits)
            
            # Get fake probability
            fake_prob = probs[0, 0].item() if probs.dim() > 1 else probs[0].item()
            real_prob = 1.0 - fake_prob
            
            return real_prob, fake_prob
    
    def batch_predict(self, input_tensor: torch.Tensor) -> List[Tuple[float, float]]:
        """
        Run prediction on batch of inputs.
        
        Args:
            input_tensor: Preprocessed images tensor [B, C, H, W]
            
        Returns:
            List of (real_probability, fake_probability) tuples
        """
        if not self.is_initialized or self.model is None:
            raise RuntimeError("Model not initialized")
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self._device)
            
            # Get logits
            logits = self.model(input_tensor)
            
            # Apply temperature scaling
            if self._temperature != 1.0:
                logits = logits / self._temperature
            
            # Binary classification
            probs = torch.sigmoid(logits).squeeze(-1)
            
            results = []
            for i in range(probs.shape[0]):
                fake_prob = probs[i].item()
                real_prob = 1.0 - fake_prob
                results.append((real_prob, fake_prob))
            
            return results
    
    def get_verdict(self, fake_probability: float) -> str:
        """
        Get verdict based on fake probability and threshold.
        
        Args:
            fake_probability: Probability that content is fake
        
        Returns:
            Verdict string: 'REAL', 'FAKE', or 'UNCERTAIN'
        """
        if fake_probability >= self._threshold + 0.15:
            return "FAKE"
        elif fake_probability <= self._threshold - 0.15:
            return "REAL"
        else:
            return "UNCERTAIN"


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


async def initialize_models() -> ModelRegistry:
    """Initialize models and return registry."""
    registry = get_model_registry()
    if not registry.is_initialized:
        await registry.initialize()
    return registry
