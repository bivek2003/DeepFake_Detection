"""
Model registry for managing detection models.
Supports demo mode with deterministic fake scoring.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    calibration_method: str | None = None
    commit_hash: str | None = None
    metrics: dict[str, Any] | None = None


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
        """
        batch_size = x.shape[0]
        
        # Use image statistics to generate pseudo-random but deterministic scores
        # This creates variation that looks realistic
        mean_values = x.mean(dim=(2, 3))  # [B, C]
        std_values = x.std(dim=(2, 3))    # [B, C]
        
        # Combine statistics into a score
        score_base = (mean_values.mean(dim=1) + std_values.mean(dim=1)) / 2
        score = torch.sigmoid(score_base - 0.5)  # Centered around 0.5
        
        # Add slight variation
        noise = torch.sin(x.sum(dim=(1, 2, 3)) * 0.001) * 0.1
        score = torch.clamp(score + noise, 0.0, 1.0)
        
        # Return as [B, 2] for softmax compatibility (real_prob, fake_prob)
        return torch.stack([1 - score, score], dim=1)


class ModelRegistry:
    """
    Registry for managing detection models.
    Handles loading, caching, and demo mode.
    """
    
    def __init__(self):
        self.model: nn.Module | None = None
        self.model_info: ModelInfo | None = None
        self.is_initialized = False
        self._device: str = "cpu"
        
        settings = get_settings()
        self._demo_mode = settings.demo_mode
        self._weights_path = Path(settings.model_weights_path)
        self._calibration_params: dict[str, float] = {}
    
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
    
    async def initialize(self) -> None:
        """Initialize model registry."""
        settings = get_settings()
        
        # Determine device
        if settings.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = settings.device
        
        logger.info(f"Using device: {self._device}")
        
        # Check for real weights
        weights_file = self._weights_path / "detector.pth"
        
        if not self._demo_mode and weights_file.exists():
            await self._load_real_model(weights_file)
        else:
            await self._load_demo_model()
        
        self.is_initialized = True
        logger.info(
            f"Model initialized",
            extra={
                "model_name": self.model_name,
                "model_version": self.model_version,
                "demo_mode": self._demo_mode,
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
            
            checkpoint = torch.load(weights_path, map_location=self._device)
            
            self.model = DeepfakeDetector()
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self._device)
            self.model.eval()
            
            # Load calibration parameters if present
            if "calibration" in checkpoint:
                self._calibration_params = checkpoint["calibration"]
            
            self.model_info = ModelInfo(
                name=checkpoint.get("model_name", "deepfake-detector"),
                version=checkpoint.get("version", "1.0.0"),
                device=self._device,
                is_demo=False,
                calibration_method=checkpoint.get("calibration_method", "temperature_scaling"),
                commit_hash=checkpoint.get("commit_hash"),
                metrics=checkpoint.get("metrics"),
            )
            
            logger.info(f"Real model loaded from {weights_path}")
            
        except Exception as e:
            logger.error(f"Failed to load real model: {e}")
            logger.info("Falling back to demo model")
            await self._load_demo_model()
    
    def predict(self, input_tensor: torch.Tensor) -> tuple[float, float]:
        """
        Run prediction on input tensor.
        
        Args:
            input_tensor: Preprocessed image tensor [B, C, H, W]
            
        Returns:
            Tuple of (real_probability, fake_probability)
        """
        if not self.is_initialized or self.model is None:
            raise RuntimeError("Model not initialized")
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self._device)
            output = self.model(input_tensor)
            
            # Apply softmax
            probs = torch.softmax(output, dim=1)
            
            # Apply temperature scaling calibration if available
            if "temperature" in self._calibration_params:
                temp = self._calibration_params["temperature"]
                probs = torch.softmax(output / temp, dim=1)
            
            # Return probabilities (real, fake)
            real_prob = probs[0, 0].item()
            fake_prob = probs[0, 1].item()
            
            return real_prob, fake_prob
    
    def batch_predict(self, input_tensor: torch.Tensor) -> list[tuple[float, float]]:
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
            output = self.model(input_tensor)
            
            # Apply softmax
            probs = torch.softmax(output, dim=1)
            
            # Apply calibration
            if "temperature" in self._calibration_params:
                temp = self._calibration_params["temperature"]
                probs = torch.softmax(output / temp, dim=1)
            
            results = []
            for i in range(probs.shape[0]):
                real_prob = probs[i, 0].item()
                fake_prob = probs[i, 1].item()
                results.append((real_prob, fake_prob))
            
            return results
