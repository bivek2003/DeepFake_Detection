"""
Video Models for Deepfake Detection

Implements EfficientNet and XceptionNet architectures as specified in the roadmap.
These are the primary video-based deepfake detection models.

EfficientNet: "EfficientNet-B7 or B4 fine-tuned on faces"
XceptionNet: "Used in DFDC baseline"

Author: Bivek Sharma Panthi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import efficientnet_b4, efficientnet_b7
import logging
from typing import Optional

from .base_model import VideoDeepfakeModel, ModelConfig, model_registry

logger = logging.getLogger(__name__)


class EfficientNetDeepfake(VideoDeepfakeModel):
    """
    EfficientNet-based deepfake detection model
    
    Implements EfficientNet-B4 or B7 as specified in the roadmap:
    "EfficientNet-B7 or B4 fine-tuned on faces often yields high accuracy"
    """
    
    def __init__(self, config: ModelConfig):
        # Set model variant from config
        self.variant = config.model_params.get('variant', 'b4')
        if self.variant not in ['b4', 'b7']:
            raise ValueError(f"EfficientNet variant must be 'b4' or 'b7', got {self.variant}")
        
        super().__init__(config)
        
        logger.info(f"Initialized EfficientNet-{self.variant.upper()} for deepfake detection")
    
    def build_backbone(self) -> nn.Module:
        """Build EfficientNet backbone"""
        if self.variant == 'b4':
            backbone = efficientnet_b4(pretrained=self.config.use_pretrained)
            self.backbone_features = 1792  # EfficientNet-B4 features
        elif self.variant == 'b7':
            backbone = efficientnet_b7(pretrained=self.config.use_pretrained)
            self.backbone_features = 2560  # EfficientNet-B7 features
        
        # Remove the classifier (keep only feature extractor)
        backbone.classifier = nn.Identity()
        
        if self.config.freeze_backbone:
            self.freeze_backbone()
        
        return backbone
    
    def build_classifier(self, backbone_features: int) -> nn.Module:
        """Build classification head with dropout and batch normalization"""
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(backbone_features),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.config.dropout_rate * 0.5),
            nn.Linear(512, self.config.num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier_weights(classifier)
        
        return classifier
    
    def get_backbone_features(self) -> int:
        """Get number of backbone features"""
        return self.backbone_features
    
    def _initialize_classifier_weights(self, classifier):
        """Initialize classifier weights with proper initialization"""
        for module in classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using EfficientNet backbone"""
        # x shape: (batch_size, 3, height, width)
        features = self.backbone.features(x)
        return features
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify features using the classification head"""
        return self.classifier(features)


class XceptionNetDeepfake(VideoDeepfakeModel):
    """
    XceptionNet-based deepfake detection model
    
    As specified in roadmap: "XceptionNet (CNN) – used in DFDC baseline"
    Implements modified Xception architecture for deepfake detection.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        logger.info("Initialized XceptionNet for deepfake detection")
    
    def build_backbone(self) -> nn.Module:
        """Build Xception-inspired backbone"""
        # Since torchvision doesn't have Xception, we implement a simplified version
        # following the Xception architecture principles
        
        backbone = XceptionBackbone(
            input_channels=3,
            use_pretrained=self.config.use_pretrained
        )
        
        self.backbone_features = 2048  # Xception output features
        
        if self.config.freeze_backbone:
            self.freeze_backbone()
        
        return backbone
    
    def build_classifier(self, backbone_features: int) -> nn.Module:
        """Build classification head"""
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(backbone_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate * 0.5),
            nn.Linear(1024, self.config.num_classes)
        )
        
        # Initialize weights
        self._initialize_weights(classifier)
        
        return classifier
    
    def get_backbone_features(self) -> int:
        """Get number of backbone features"""
        return self.backbone_features
    
    def _initialize_weights(self, module):
        """Initialize weights following Xception paper"""
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class XceptionBackbone(nn.Module):
    """
    Simplified Xception backbone for deepfake detection
    
    Implements key Xception principles:
    - Depthwise separable convolutions
    - Skip connections
    - Entry, Middle, and Exit flows
    """
    
    def __init__(self, input_channels: int = 3, use_pretrained: bool = False):
        super().__init__()
        
        # Entry flow
        self.entry_flow = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 2 - Separable Conv
            SeparableConv2d(64, 128, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 3 - Separable Conv
            SeparableConv2d(128, 256, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 4 - Separable Conv
            SeparableConv2d(256, 728, stride=2),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
        )
        
        # Middle flow (8 blocks)
        self.middle_flow = nn.ModuleList([
            XceptionBlock(728, 728) for _ in range(8)
        ])
        
        # Exit flow
        self.exit_flow = nn.Sequential(
            SeparableConv2d(728, 1024, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            SeparableConv2d(1024, 1536, stride=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            
            SeparableConv2d(1536, 2048, stride=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """Forward pass through Xception backbone"""
        # Entry flow
        x = self.entry_flow(x)
        
        # Middle flow with residual connections
        for block in self.middle_flow:
            x = block(x)
        
        # Exit flow
        x = self.exit_flow(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize weights following Xception paper"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution as used in Xception"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=bias
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """Xception block with residual connection"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.separable_conv1 = SeparableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.separable_conv2 = SeparableConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.separable_conv3 = SeparableConv2d(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.relu1(self.bn1(self.separable_conv1(x)))
        out = self.relu2(self.bn2(self.separable_conv2(out)))
        out = self.bn3(self.separable_conv3(out))
        
        out += residual
        return F.relu(out, inplace=True)


class MultiScaleVideoModel(VideoDeepfakeModel):
    """
    Multi-scale video model for enhanced deepfake detection
    
    Processes input at multiple scales to capture both fine-grained 
    and global artifacts commonly found in deepfakes.
    """
    
    def __init__(self, config: ModelConfig):
        self.scales = config.model_params.get('scales', [1.0, 0.75, 0.5])
        self.backbone_type = config.model_params.get('backbone', 'efficientnet_b4')
        
        super().__init__(config)
        logger.info(f"Initialized Multi-scale model with {len(self.scales)} scales")
    
    def build_backbone(self) -> nn.Module:
        """Build multi-scale backbone"""
        backbones = nn.ModuleDict()
        
        # Create backbone for each scale
        for i, scale in enumerate(self.scales):
            if self.backbone_type == 'efficientnet_b4':
                backbone = efficientnet_b4(pretrained=self.config.use_pretrained)
                backbone.classifier = nn.Identity()
                self.backbone_features = 1792
            else:
                raise ValueError(f"Unsupported backbone: {self.backbone_type}")
            
            backbones[f'scale_{i}'] = backbone
        
        return backbones
    
    def build_classifier(self, backbone_features: int) -> nn.Module:
        """Build fusion classifier for multi-scale features"""
        # Total features = backbone_features * num_scales
        total_features = backbone_features * len(self.scales)
        
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # Feature fusion layer
            nn.Linear(total_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(self.config.dropout_rate),
            
            # Classification layers
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout_rate * 0.5),
            nn.Linear(512, self.config.num_classes)
        )
        
        return classifier
    
    def get_backbone_features(self) -> int:
        """Get combined backbone features"""
        return self.backbone_features * len(self.scales)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale features"""
        batch_size = x.size(0)
        multi_scale_features = []
        
        for i, scale in enumerate(self.scales):
            if scale != 1.0:
                # Resize input for different scales
                h, w = x.size(2), x.size(3)
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                scaled_x = x
            
            # Extract features at this scale
            backbone = self.backbone[f'scale_{i}']
            features = backbone.features(scaled_x)
            
            # Global average pooling to get fixed size features
            pooled_features = F.adaptive_avg_pool2d(features, (1, 1))
            pooled_features = pooled_features.view(batch_size, -1)
            
            multi_scale_features.append(pooled_features)
        
        # Concatenate all scale features
        combined_features = torch.cat(multi_scale_features, dim=1)
        return combined_features.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions for classifier
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify combined multi-scale features"""
        return self.classifier(features)


# Register models with the global registry
def register_video_models():
    """Register all video models with the model registry"""
    
    # EfficientNet-B4
    efficientnet_b4_config = ModelConfig(
        model_name="efficientnet_b4",
        input_size=(224, 224),
        model_params={'variant': 'b4'}
    )
    model_registry.register_model("efficientnet_b4", EfficientNetDeepfake, efficientnet_b4_config)
    
    # EfficientNet-B7 (for larger datasets/more compute)
    efficientnet_b7_config = ModelConfig(
        model_name="efficientnet_b7",
        input_size=(224, 224),
        model_params={'variant': 'b7'}
    )
    model_registry.register_model("efficientnet_b7", EfficientNetDeepfake, efficientnet_b7_config)
    
    # XceptionNet
    xception_config = ModelConfig(
        model_name="xception",
        input_size=(224, 224),
        dropout_rate=0.3  # Higher dropout for Xception
    )
    model_registry.register_model("xception", XceptionNetDeepfake, xception_config)
    
    # Multi-scale model
    multiscale_config = ModelConfig(
        model_name="multiscale_efficientnet",
        input_size=(224, 224),
        model_params={
            'scales': [1.0, 0.75, 0.5],
            'backbone': 'efficientnet_b4'
        }
    )
    model_registry.register_model("multiscale_efficientnet", MultiScaleVideoModel, multiscale_config)
    
    logger.info("Registered 4 video models: EfficientNet-B4/B7, XceptionNet, Multi-scale")


# Auto-register models when module is imported
register_video_models()


def create_efficientnet_model(variant: str = 'b4', 
                             input_size: tuple = (224, 224),
                             num_classes: int = 2,
                             pretrained: bool = True) -> EfficientNetDeepfake:
    """Convenience function to create EfficientNet model"""
    config = ModelConfig(
        model_name=f"efficientnet_{variant}",
        input_size=input_size,
        num_classes=num_classes,
        use_pretrained=pretrained,
        model_params={'variant': variant}
    )
    return EfficientNetDeepfake(config)


def create_xception_model(input_size: tuple = (224, 224),
                         num_classes: int = 2,
                         pretrained: bool = True) -> XceptionNetDeepfake:
    """Convenience function to create XceptionNet model"""
    config = ModelConfig(
        model_name="xception",
        input_size=input_size,
        num_classes=num_classes,
        use_pretrained=pretrained,
        dropout_rate=0.3
    )
    return XceptionNetDeepfake(config)


def main():
    """Demonstrate video models"""
    print("🎬 VIDEO MODELS DEMO")
    print("=" * 50)
    
    # Test EfficientNet-B4
    print("\n🔍 Testing EfficientNet-B4...")
    efficientnet_model = create_efficientnet_model('b4')
    
    # Test input
    test_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    with torch.no_grad():
        output = efficientnet_model(test_input, return_features=True)
        print(f"  ✓ Output shape: {output.logits.shape}")
        print(f"  ✓ Probabilities: {output.probabilities[0].cpu().numpy()}")
        print(f"  ✓ Model params: {sum(p.numel() for p in efficientnet_model.parameters()):,}")
    
    # Test XceptionNet
    print("\n🔍 Testing XceptionNet...")
    xception_model = create_xception_model()
    
    with torch.no_grad():
        output = xception_model(test_input, return_features=True)
        print(f"  ✓ Output shape: {output.logits.shape}")
        print(f"  ✓ Probabilities: {output.probabilities[0].cpu().numpy()}")
        print(f"  ✓ Model params: {sum(p.numel() for p in xception_model.parameters()):,}")
    
    # Show model info
    print(f"\n📊 Model Information:")
    efficientnet_info = efficientnet_model.get_model_info()
    print(f"  EfficientNet-B4:")
    print(f"    - Parameters: {efficientnet_info['total_parameters']:,}")
    print(f"    - Model size: {efficientnet_info['model_size_mb']:.1f} MB")
    
    xception_info = xception_model.get_model_info()
    print(f"  XceptionNet:")
    print(f"    - Parameters: {xception_info['total_parameters']:,}")
    print(f"    - Model size: {xception_info['model_size_mb']:.1f} MB")
    
    print(f"\n✅ Video models ready for training!")


if __name__ == "__main__":
    main()
