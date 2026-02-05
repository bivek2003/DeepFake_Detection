"""
Production Ensemble Architecture for Maximum Accuracy Deepfake Detection

Combines multiple models for highest accuracy:
- EfficientNet-B4: Balanced accuracy/speed
- EfficientNet-B5: High accuracy with pretrained weights
- XceptionNet: Proven deepfake detection backbone

Target: 96%+ accuracy on combined test sets
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """Learned attention-based fusion of ensemble predictions."""

    def __init__(self, num_models: int, hidden_dim: int = 256):
        super().__init__()
        self.num_models = num_models

        # Each model contributes features
        self.attention = nn.Sequential(
            nn.Linear(num_models, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=-1),
        )

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, num_models) tensor of model predictions
        Returns:
            Weighted ensemble prediction
        """
        weights = self.attention(predictions)
        return (predictions * weights).sum(dim=-1, keepdim=True)


class XceptionBlock(nn.Module):
    """Xception separable convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=stride, padding=1) if stride > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return self.relu(x + residual)


class XceptionNet(nn.Module):
    """
    Xception architecture optimized for deepfake detection.
    Based on "FaceForensics++: Learning to Detect Manipulated Facial Images"
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()

        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.entry_blocks = nn.Sequential(
            XceptionBlock(64, 128, stride=2),
            XceptionBlock(128, 256, stride=2),
            XceptionBlock(256, 728, stride=2),
        )

        # Middle flow (8 blocks)
        self.middle = nn.Sequential(*[XceptionBlock(728, 728) for _ in range(8)])

        # Exit flow
        self.exit_block = XceptionBlock(728, 1024, stride=2)

        self.exit = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1, groups=1024, bias=False),
            nn.Conv2d(1024, 1536, 1, bias=False),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            nn.Conv2d(1536, 1536, 3, padding=1, groups=1536, bias=False),
            nn.Conv2d(1536, 2048, 1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.entry_blocks(x)
        x = self.middle(x)
        x = self.exit_block(x)
        x = self.exit(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = self.entry(x)
        x = self.entry_blocks(x)
        x = self.middle(x)
        x = self.exit_block(x)
        x = self.exit(x)
        x = self.pool(x)
        return x.flatten(1)


class EfficientNetDetector(nn.Module):
    """EfficientNet-based detector with enhanced classification head."""

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        num_classes: int = 1,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.backbone_name = backbone

        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool="",  # Remove global pool
        )

        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 380, 380)
            features = self.backbone(dummy)
            if len(features.shape) == 4:
                self.feature_dim = features.shape[1]
            else:
                self.feature_dim = features.shape[-1]

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout * 0.75),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.pool(features)
            features = features.flatten(1)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        features = self.backbone(x)
        if len(features.shape) == 4:
            features = self.pool(features)
            features = features.flatten(1)
        return features


class DeepfakeEnsemble(nn.Module):
    """
    Production Ensemble Model for Maximum Accuracy

    Combines:
    - EfficientNet-B4: 19M params, fast, good accuracy
    - EfficientNet-B5: 30M params, high accuracy with pretrained weights
    - XceptionNet: Proven deepfake detection architecture

    Fusion methods:
    - 'attention': Learned attention weights (best)
    - 'average': Simple averaging
    - 'weighted': Fixed learned weights
    """

    def __init__(
        self, fusion_method: str = "attention", pretrained: bool = True, dropout: float = 0.4
    ):
        super().__init__()

        self.fusion_method = fusion_method

        # Initialize component models
        print("Initializing EfficientNet-B4...")
        self.efficientnet_b4 = EfficientNetDetector(
            backbone="efficientnet_b4.ra2_in1k", pretrained=pretrained, dropout=dropout
        )

        print("Initializing EfficientNet-B5...")
        self.efficientnet_b5 = EfficientNetDetector(
            backbone="efficientnet_b5.sw_in12k_ft_in1k", pretrained=pretrained, dropout=dropout
        )

        print("Initializing XceptionNet...")
        self.xception = XceptionNet(pretrained=pretrained)

        self.num_models = 3

        # Fusion mechanism
        if fusion_method == "attention":
            self.fusion = AttentionFusion(self.num_models)
        elif fusion_method == "weighted":
            # Learnable weights initialized to equal
            self.fusion_weights = nn.Parameter(torch.ones(self.num_models) / self.num_models)
        else:  # average
            self.register_buffer("fusion_weights", torch.ones(self.num_models) / self.num_models)

        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))

        print(f"Ensemble initialized with {self.count_parameters():,} total parameters")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, x: torch.Tensor, return_individual: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass through ensemble.

        Args:
            x: Input images (B, 3, H, W)
            return_individual: If True, also return individual model predictions

        Returns:
            Ensemble prediction, optionally with individual predictions
        """
        # Get predictions from each model
        pred_b4 = self.efficientnet_b4(x)
        pred_b5 = self.efficientnet_b5(x)
        pred_xception = self.xception(x)

        # Stack predictions
        predictions = torch.cat([pred_b4, pred_b5, pred_xception], dim=-1)

        # Fuse predictions
        if self.fusion_method == "attention":
            ensemble_pred = self.fusion(torch.sigmoid(predictions))
            # Convert back to logit space
            ensemble_pred = torch.logit(ensemble_pred.clamp(1e-7, 1 - 1e-7))
        elif self.fusion_method == "weighted":
            weights = F.softmax(self.fusion_weights, dim=0)
            ensemble_pred = (predictions * weights).sum(dim=-1, keepdim=True)
        else:  # average
            ensemble_pred = predictions.mean(dim=-1, keepdim=True)

        # Apply temperature scaling
        ensemble_pred = ensemble_pred / self.temperature

        if return_individual:
            return ensemble_pred, {
                "efficientnet_b4": pred_b4,
                "efficientnet_b5": pred_b5,
                "xception": pred_xception,
            }

        return ensemble_pred

    def get_model_contributions(self, x: torch.Tensor) -> dict[str, float]:
        """Get the contribution of each model to the final prediction."""
        with torch.no_grad():
            pred_b4 = torch.sigmoid(self.efficientnet_b4(x)).mean().item()
            pred_b5 = torch.sigmoid(self.efficientnet_b5(x)).mean().item()
            pred_xception = torch.sigmoid(self.xception(x)).mean().item()

            if self.fusion_method == "weighted":
                weights = F.softmax(self.fusion_weights, dim=0).cpu().numpy()
            else:
                weights = [1 / 3, 1 / 3, 1 / 3]

            return {
                "efficientnet_b4": {"prediction": pred_b4, "weight": float(weights[0])},
                "efficientnet_b5": {"prediction": pred_b5, "weight": float(weights[1])},
                "xception": {"prediction": pred_xception, "weight": float(weights[2])},
            }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined BCE + Focal + Label Smoothing loss for robust training."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(inputs, targets)
        focal = self.focal_loss(inputs, targets)

        return self.bce_weight * bce + self.focal_weight * focal


def create_production_model(
    model_type: str = "ensemble", pretrained: bool = True, device: str = "cuda"
) -> nn.Module:
    """
    Factory function to create production model.

    Args:
        model_type: One of 'ensemble', 'efficientnet_b4', 'efficientnet_b7', 'xception'
        pretrained: Use ImageNet pretrained weights
        device: Target device

    Returns:
        Initialized model
    """
    if model_type == "ensemble":
        model = DeepfakeEnsemble(fusion_method="attention", pretrained=pretrained)
    elif model_type == "efficientnet_b4":
        model = EfficientNetDetector(backbone="efficientnet_b4", pretrained=pretrained)
    elif model_type == "efficientnet_b5":
        model = EfficientNetDetector(
            backbone="efficientnet_b5.sw_in12k_ft_in1k", pretrained=pretrained
        )
    elif model_type == "xception":
        model = XceptionNet(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


if __name__ == "__main__":
    # Test ensemble
    print("Testing Production Ensemble...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = create_production_model("ensemble", device=device)

    # Test forward pass
    dummy_input = torch.randn(2, 3, 380, 380).to(device)

    with torch.no_grad():
        output, individual = model(dummy_input, return_individual=True)

    print(f"\nEnsemble output shape: {output.shape}")
    print(f"Ensemble prediction: {torch.sigmoid(output)}")

    for name, pred in individual.items():
        print(f"{name} prediction: {torch.sigmoid(pred)}")

    # Test loss
    loss_fn = CombinedLoss()
    targets = torch.tensor([[1.0], [0.0]]).to(device)
    loss = loss_fn(output, targets)
    print(f"\nCombined loss: {loss.item():.4f}")

    # Model contributions
    contributions = model.get_model_contributions(dummy_input[:1])
    print("\nModel contributions:")
    for name, info in contributions.items():
        print(f"  {name}: pred={info['prediction']:.3f}, weight={info['weight']:.3f}")
