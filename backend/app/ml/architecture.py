"""
Deep Learning Model Architecture for Deepfake Detection

EfficientNet-B4 based binary classifier with attention mechanisms.
Designed for high accuracy deepfake detection with efficient inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import math

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not installed. Some features will be unavailable.")


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer for feature aggregation.
    """
    
    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) or (batch, seq, features)
        if x.dim() == 2:
            return x
        
        # Compute attention weights
        weights = self.attention(x)  # (batch, seq, 1)
        weights = F.softmax(weights, dim=1)
        
        # Weighted sum
        output = (x * weights).sum(dim=1)
        
        return output


class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B4 based deepfake detection model.
    
    Architecture:
    - EfficientNet-B4 backbone (pretrained on ImageNet)
    - Dropout for regularization
    - Multi-layer classification head
    - Optional attention pooling
    
    Input: RGB images of size 380x380
    Output: Binary classification (0=Real, 1=Fake)
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout_rate: float = 0.5,
        hidden_size: int = 512,
        use_attention: bool = False,
    ):
        """
        Initialize the deepfake detector.
        
        Args:
            backbone: Backbone architecture name (timm model)
            pretrained: Whether to use ImageNet pretrained weights
            num_classes: Number of output classes (1 for binary)
            dropout_rate: Dropout rate in classification head
            hidden_size: Hidden layer size in classification head
            use_attention: Whether to use attention pooling
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for model architecture")
        
        # Create backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='',  # Remove global pooling
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 380, 380)
            features = self.backbone(dummy_input)
            if features.dim() == 4:
                # (batch, channels, h, w)
                self.feature_dim = features.shape[1]
                self.spatial_dim = features.shape[2] * features.shape[3]
            else:
                self.feature_dim = features.shape[-1]
                self.spatial_dim = 1
        
        print(f"Backbone: {backbone}, Feature dim: {self.feature_dim}")
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Optional attention
        if use_attention:
            self.attention_pool = AttentionPooling(self.feature_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.6),  # Reduced dropout in second layer
            nn.Linear(hidden_size, num_classes),
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 3, H, W)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        if features.dim() == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
        
        Returns:
            Probabilities of shape (batch, 1) for binary classification
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.
        
        Args:
            x: Input tensor
        
        Returns:
            Features of shape (batch, feature_dim)
        """
        features = self.backbone(x)
        
        if features.dim() == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        
        return features


class DeepfakeDetectorEnsemble(nn.Module):
    """
    Ensemble of multiple deepfake detectors.
    
    Combines predictions from multiple models for improved robustness.
    """
    
    def __init__(
        self,
        backbones: list = ["efficientnet_b4", "efficientnet_b3"],
        pretrained: bool = True,
        ensemble_method: str = "average",
    ):
        """
        Initialize ensemble.
        
        Args:
            backbones: List of backbone architectures
            pretrained: Whether to use pretrained weights
            ensemble_method: How to combine predictions ('average', 'max', 'voting')
        """
        super().__init__()
        
        self.ensemble_method = ensemble_method
        
        self.models = nn.ModuleList([
            DeepfakeDetector(backbone=bb, pretrained=pretrained)
            for bb in backbones
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ensemble aggregation."""
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=0)  # (num_models, batch, 1)
        
        if self.ensemble_method == "average":
            return outputs.mean(dim=0)
        elif self.ensemble_method == "max":
            return outputs.max(dim=0)[0]
        else:
            # Voting (threshold at 0.5)
            votes = (torch.sigmoid(outputs) > 0.5).float()
            return votes.mean(dim=0) * 2 - 1  # Scale to logit-like range


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape (batch, 1)
            targets: Labels of shape (batch,) with values 0 or 1
        
        Returns:
            Loss value
        """
        inputs = inputs.view(-1)
        targets = targets.float().view(-1)
        
        # Compute BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        
        # Compute focal weight
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Combined BCE + Focal Loss for training.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce_loss = self.bce(inputs.view(-1), targets.float())
        focal_loss = self.focal(inputs, targets)
        
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss


def create_model(
    backbone: str = "efficientnet_b4",
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
) -> DeepfakeDetector:
    """
    Factory function to create a model.
    
    Args:
        backbone: Backbone architecture
        pretrained: Use ImageNet pretrained weights
        checkpoint_path: Path to trained checkpoint
        device: Device to load model on
    
    Returns:
        DeepfakeDetector model
    """
    model = DeepfakeDetector(
        backbone=backbone,
        pretrained=pretrained if checkpoint_path is None else False,
    )
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    
    return model


# Model configurations for different use cases
MODEL_CONFIGS = {
    "efficientnet_b4": {
        "backbone": "efficientnet_b4",
        "image_size": 380,
        "dropout": 0.5,
        "description": "Best accuracy/speed tradeoff for production",
    },
    "efficientnet_b3": {
        "backbone": "efficientnet_b3",
        "image_size": 300,
        "dropout": 0.4,
        "description": "Faster inference with good accuracy",
    },
    "efficientnet_b5": {
        "backbone": "efficientnet_b5",
        "image_size": 456,
        "dropout": 0.5,
        "description": "Higher accuracy, requires more memory",
    },
    "resnet50": {
        "backbone": "resnet50",
        "image_size": 224,
        "dropout": 0.5,
        "description": "Classic architecture, fast inference",
    },
}
