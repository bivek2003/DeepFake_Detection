"""
AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention) model
for audio deepfake detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

try:
    from .base_model import BaseDeepfakeModel, ModelConfig
except ImportError:
    # Fallback base class
    class BaseDeepfakeModel(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config
    
    class ModelConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

# Check if transformers is available
try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("transformers not available, using simple CNN frontend")

class SpectrogramFrontend(nn.Module):
    """Spectrogram-based frontend for audio processing."""
    
    def __init__(self, n_fft=512, hop_length=256, n_mels=80):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Mel filterbank (simplified)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (batch, 1, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        return x  # (batch, 256, time_reduced)

class GraphAttentionLayer(nn.Module):
    """Graph attention layer for spectro-temporal modeling."""
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, h):
        # h: (batch, seq_len, in_features)
        Wh = self.W(h)  # (batch, seq_len, out_features)
        
        # Compute attention
        batch_size, seq_len, _ = Wh.size()
        
        # Create all pairs for attention computation
        a_input = torch.cat([
            Wh.unsqueeze(2).expand(-1, -1, seq_len, -1),  # (batch, seq_len, seq_len, out_features)
            Wh.unsqueeze(1).expand(-1, seq_len, -1, -1)   # (batch, seq_len, seq_len, out_features)
        ], dim=3)  # (batch, seq_len, seq_len, 2*out_features)
        
        e = self.leakyrelu(self.a(a_input).squeeze(3))  # (batch, seq_len, seq_len)
        
        attention = F.softmax(e, dim=2)
        attention = self.dropout_layer(attention)
        
        h_prime = torch.matmul(attention, Wh)  # (batch, seq_len, out_features)
        
        return h_prime

class AASISTDeepfake(BaseDeepfakeModel):
    """AASIST model for audio deepfake detection."""
    
    def __init__(self, 
                 num_classes: int = 2,
                 dropout_rate: float = 0.1,
                 hidden_dim: int = 256,
                 num_attention_heads: int = 4,
                 config: Optional[ModelConfig] = None):
        """
        Initialize AASIST model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate
            hidden_dim: Hidden dimension
            num_attention_heads: Number of attention heads
            config: Model configuration
        """
        super().__init__(config)
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        
        # Frontend: either Wav2Vec2 or spectrogram
        if HAS_TRANSFORMERS:
            try:
                # Try to use a small Wav2Vec2 model
                wav2vec_config = Wav2Vec2Config(
                    hidden_size=256,
                    num_attention_heads=4,
                    num_hidden_layers=4,
                    vocab_size=32,  # Not used for feature extraction
                )
                self.frontend = Wav2Vec2Model(wav2vec_config)
                self.frontend_dim = 256
            except Exception:
                # Fallback to spectrogram frontend
                self.frontend = SpectrogramFrontend()
                self.frontend_dim = 256
        else:
            # Use spectrogram frontend
            self.frontend = SpectrogramFrontend()
            self.frontend_dim = 256
        
        # Spectro-temporal graph attention
        self.graph_attention_layers = nn.ModuleList([
            GraphAttentionLayer(self.frontend_dim, hidden_dim, dropout_rate)
            for _ in range(num_attention_heads)
        ])
        
        # Temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim * num_attention_heads,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from audio input."""
        # x: (batch, 1, time) or (batch, time)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Frontend processing
        if hasattr(self.frontend, 'feature_extractor'):
            # Wav2Vec2 frontend
            if len(x.shape) == 3:
                x = x.squeeze(1)  # Remove channel for Wav2Vec2
            features = self.frontend(x).last_hidden_state
        else:
            # Spectrogram frontend
            features = self.frontend(x)  # (batch, channels, time)
            features = features.transpose(1, 2)  # (batch, time, channels)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.extract_features(x)  # (batch, time, features)
        
        # Graph attention
        attention_outputs = []
        for attention_layer in self.graph_attention_layers:
            att_out = attention_layer(features)
            attention_outputs.append(att_out)
        
        # Concatenate attention outputs
        graph_features = torch.cat(attention_outputs, dim=-1)  # (batch, time, hidden_dim * num_heads)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(graph_features)  # (batch, time, hidden_dim * 2)
        
        # Global pooling over time
        pooled = lstm_out.transpose(1, 2)  # (batch, hidden_dim * 2, time)
        pooled = self.global_pool(pooled)  # (batch, hidden_dim * 2, 1)
        pooled = pooled.squeeze(-1)  # (batch, hidden_dim * 2)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'hidden_dim': self.hidden_dim,
            'num_attention_heads': self.num_attention_heads,
        }

def create_aasist_model(num_classes: int = 2,
                       dropout_rate: float = 0.1,
                       hidden_dim: int = 256,
                       num_attention_heads: int = 4,
                       **kwargs) -> AASISTDeepfake:
    """
    Create an AASIST model for audio deepfake detection.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        hidden_dim: Hidden dimension
        num_attention_heads: Number of attention heads
        **kwargs: Additional arguments
    
    Returns:
        AASISTDeepfake model
    """
    config = ModelConfig(
        model_type="aasist",
        num_classes=num_classes,
        input_size=(1, 64000),  # 4 seconds at 16kHz
        **kwargs
    )
    
    return AASISTDeepfake(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads,
        config=config
    )

__all__ = [
    'AASISTDeepfake',
    'create_aasist_model',
    'SpectrogramFrontend',
    'GraphAttentionLayer'
]
