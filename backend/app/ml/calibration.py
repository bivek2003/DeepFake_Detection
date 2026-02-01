"""
Model calibration utilities.
Implements temperature scaling for probability calibration.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS

from app.logging_config import get_logger

logger = get_logger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    Learns a single temperature parameter to scale logits.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """
        Fit temperature parameter using validation set.
        
        Args:
            logits: Model logits [N, num_classes]
            labels: True labels [N]
            max_iter: Maximum iterations for optimization
            
        Returns:
            Optimal temperature value
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        logger.info(f"Calibration complete. Temperature: {self.temperature.item():.4f}")
        
        return self.temperature.item()


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities for positive class [N]
        labels: True binary labels [N]
        n_bins: Number of bins for calibration
        
    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            ece += (bin_size / len(probs)) * np.abs(avg_confidence - avg_accuracy)
    
    return ece


def apply_temperature_scaling(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    
    Args:
        logits: Raw model logits
        temperature: Temperature parameter
        
    Returns:
        Scaled probabilities
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)
