"""
Audio Models for Deepfake Detection

Implements AASIST and Wav2Vec2+AASIST architectures as specified in the roadmap.
These are state-of-the-art models for audio deepfake detection.

AASIST: "Audio Anti-Spoofing Integrated Softmax - graph-attention model proven for spoof detection"
Wav2Vec2+AASIST: "Wav2Vec2.0 + AASIST (W2V+AASIST) combination"

Author: Your Name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
import logging
from typing import Optional, Tuple
import numpy as np

from ..base_model import AudioDeepfakeModel, ModelConfig, model_registry

logger = logging.getLogger(__name__)


class AASIST(AudioDeepfakeModel):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
    
    As specified in roadmap: "AASIST (Audio Anti-Spoofing Integrated Softmax) - 
    graph-attention model proven for spoof detection"
    
    Implementation based on the AASIST paper for audio anti-spoofing.
    """
    
    def __init__(self, config: ModelConfig):
        # AASIST specific parameters
        self.d_args = config.model_params.get('d_args', {
            'nb_samp': 48000,  # 3 seconds at 16kHz
            'first_conv': 128,
            'filts': [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            'gat_dims': [64, 32],
            'pool_ratios': [0.5, 0.7, 0.5, 0.5]
        })
        
        super().__init__(config)
        logger.info("Initialized AASIST model for audio anti-spoofing")
    
    def build_backbone(self) -> nn.Module:
        """Build AASIST backbone with spectro-temporal processing"""
        return AASISTBackbone(self.d_args)
    
    def build_classifier(self, backbone_features: int) -> nn.Module:
        """Build AASIST classifier with attention pooling"""
        classifier = nn.Sequential(
            nn.Linear(backbone_features, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, self.config.num_classes)
        )
        return classifier
    
    def get_backbone_features(self) -> int:
        """Get AASIST backbone features"""
        return self.d_args['gat_dims'][-1]
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using AASIST backbone"""
        # x shape: (batch_size, time_samples) for raw audio
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        return self.backbone(x)
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify using AASIST head"""
        return self.classifier(features)


class AASISTBackbone(nn.Module):
    """AASIST backbone implementing spectro-temporal graph attention"""
    
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        
        # Spectro-temporal front-end
        self.conv_time = nn.Conv1d(1, d_args['first_conv'], 3, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_ch = d_args['first_conv']
        
        for filt in d_args['filts']:
            if isinstance(filt, int):
                # Time convolution
                self.res_blocks.append(
                    ResBlock(in_ch, filt, 'time')
                )
                in_ch = filt
            else:
                # Frequency convolution  
                out_ch1, out_ch2 = filt
                self.res_blocks.append(
                    ResBlock(in_ch, out_ch1, 'freq')
                )
                self.res_blocks.append(
                    ResBlock(out_ch1, out_ch2, 'freq')
                )
                in_ch = out_ch2
        
        # Graph Attention Networks
        self.gat_layers = nn.ModuleList()
        gat_in = in_ch
        for gat_dim in d_args['gat_dims']:
            self.gat_layers.append(GraphAttentionLayer(gat_in, gat_dim))
            gat_in = gat_dim
        
        # Attention pooling
        self.attention_pool = AttentionPooling(d_args['gat_dims'][-1])
        
    def forward(self, x):
        """Forward pass through AASIST backbone"""
        # x shape: (batch, 1, time)
        
        # Time convolution
        x = self.conv_time(x)  # (batch, first_conv, time)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Convert to spectro-temporal representation
        x = self.to_spectro_temporal(x)
        
        # Apply graph attention layers
        for gat in self.gat_layers:
            x = gat(x)
        
        # Attention pooling
        x = self.attention_pool(x)
        
        return x
    
    def to_spectro_temporal(self, x):
        """Convert to spectro-temporal representation for GAT processing"""
        # Simple approach: treat each time step as a node
        # x shape: (batch, channels, time)
        batch_size, channels, time_steps = x.shape
        
        # Transpose to (batch, time, channels) for GAT processing
        x = x.transpose(1, 2)  # (batch, time, channels)
        
        return x


class ResBlock(nn.Module):
    """Residual block for AASIST"""
    
    def __init__(self, in_channels, out_channels, conv_type='time'):
        super().__init__()
        self.conv_type = conv_type
        
        if conv_type == 'time':
            self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
        else:  # freq
            # For frequency processing, we use 2D convs
            # Treat channels as frequency bins
            self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        return F.relu(out)


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for AASIST"""
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformations
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h):
        """
        Forward pass of graph attention
        h: (batch_size, num_nodes, in_features)
        """
        batch_size, num_nodes, _ = h.shape
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (batch, num_nodes, out_features)
        
        # Attention mechanism
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # Create adjacency matrix (fully connected for simplicity)
        attention = F.softmax(e, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention.unsqueeze(-1) * Wh.unsqueeze(1), 
                              torch.ones(num_nodes, 1).to(h.device)).squeeze(-1)
        
        return F.elu(h_prime)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """Prepare input for attention mechanism"""
        batch_size, num_nodes, out_features = Wh.shape
        
        # Create all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(num_nodes, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, num_nodes, 1)
        
        # Concatenate to get all combinations
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        
        return all_combinations_matrix.view(batch_size, num_nodes, num_nodes, 2 * out_features)


class AttentionPooling(nn.Module):
    """Attention-based pooling layer"""
    
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, features)
        returns: (batch_size, features)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        output = torch.sum(attn_weights * x, dim=1)
        return output


class Wav2VecAASIST(AudioDeepfakeModel):
    """
    Wav2Vec2.0 + AASIST combination model
    
    As specified in roadmap: "Wav2Vec2.0 + AASIST (W2V+AASIST) combination"
    Uses Wav2Vec2.0 as feature extractor with AASIST classifier.
    """
    
    def __init__(self, config: ModelConfig):
        self.wav2vec_model = config.model_params.get('wav2vec_model', 'facebook/wav2vec2-base')
        self.freeze_wav2vec = config.model_params.get('freeze_wav2vec', True)
        
        super().__init__(config)
        logger.info(f"Initialized Wav2Vec2.0 + AASIST with model: {self.wav2vec_model}")
    
    def build_backbone(self) -> nn.Module:
        """Build Wav2Vec2.0 + AASIST backbone"""
        return Wav2VecAASISTBackbone(
            wav2vec_model=self.wav2vec_model,
            freeze_wav2vec=self.freeze_wav2vec,
            aasist_dims=[256, 128]
        )
    
    def build_classifier(self, backbone_features: int) -> nn.Module:
        """Build classifier for combined features"""
        classifier = nn.Sequential(
            nn.Linear(backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate * 0.5),
            nn.Linear(128, self.config.num_classes)
        )
        return classifier
    
    def get_backbone_features(self) -> int:
        """Get combined backbone features"""
        return 128  # AASIST output dimension
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract combined Wav2Vec2 + AASIST features"""
        return self.backbone(x)
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify combined features"""
        return self.classifier(features)


class Wav2VecAASISTBackbone(nn.Module):
    """Combined Wav2Vec2.0 and AASIST backend"""
    
    def __init__(self, wav2vec_model='facebook/wav2vec2-base', 
                 freeze_wav2vec=True, aasist_dims=[256, 128]):
        super().__init__()
        
        try:
            # Try to import transformers for Wav2Vec2
            from transformers import Wav2Vec2Model
            self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model)
            self.wav2vec_features = self.wav2vec.config.hidden_size
            
            if freeze_wav2vec:
                for param in self.wav2vec.parameters():
                    param.requires_grad = False
                    
        except ImportError:
            logger.warning("transformers not available, using simple CNN frontend")
            # Fallback to simple CNN
            self.wav2vec = SimpleCNNFrontend()
            self.wav2vec_features = 768  # Match wav2vec2-base
        
        # AASIST processing layers
        self.aasist_layers = nn.ModuleList()
        in_dim = self.wav2vec_features
        for dim in aasist_dims:
            self.aasist_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, dim),
                    nn.ReLU(),
                    nn.LayerNorm(dim),
                    nn.Dropout(0.1)
                )
            )
            in_dim = dim
        
        # Final attention pooling
        self.attention_pool = AttentionPooling(aasist_dims[-1])
    
    def forward(self, x):
        """Forward pass through combined model"""
        # x shape: (batch, time) or (batch, 1, time)
        if x.dim() == 3:
            x = x.squeeze(1)  # Remove channel dimension for Wav2Vec2
        
        # Wav2Vec2 feature extraction
        if hasattr(self.wav2vec, 'extract_features'):
            wav2vec_features = self.wav2vec(x).last_hidden_state
        else:
            wav2vec_features = self.wav2vec(x)
        
        # Apply AASIST layers
        features = wav2vec_features
        for layer in self.aasist_layers:
            features = layer(features)
        
        # Attention pooling
        pooled_features = self.attention_pool(features)
        
        return pooled_features


class SimpleCNNFrontend(nn.Module):
    """Simple CNN frontend as fallback for Wav2Vec2"""
    
    def __init__(self):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(512, 768, kernel_size=4, stride=2),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """Forward pass through CNN frontend"""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        x = self.frontend(x)
        x = x.transpose(1, 2)  # (batch, time, features)
        return x


class RawNetDeepfake(AudioDeepfakeModel):
    """
    RawNet2-based deepfake detection model
    
    As specified in roadmap: "RawNet2 / LCNN - other top ASVspoof models 
    (raw waveform/LCNN) for deepfake audio"
    
    Processes raw audio waveforms directly without spectral features.
    """
    
    def __init__(self, config: ModelConfig):
        # RawNet specific parameters
        self.rawnet_params = config.model_params.get('rawnet_params', {
            'nb_samp': 48000,  # 3 seconds at 16kHz
            'first_conv': 128,
            'in_channels': 1,
            'filts': [128, [128, 128], [128, 256], [256, 256]],
            'blocks': [2, 4, 4, 4],
            'nb_fc_node': 1024,
            'gru_node': 1024
        })
        
        super().__init__(config)
        logger.info("Initialized RawNet2 model for raw audio processing")
    
    def build_backbone(self) -> nn.Module:
        """Build RawNet2 backbone"""
        return RawNetBackbone(self.rawnet_params)
    
    def build_classifier(self, backbone_features: int) -> nn.Module:
        """Build RawNet classifier"""
        classifier = nn.Sequential(
            nn.Linear(backbone_features, 512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, self.config.num_classes)
        )
        return classifier
    
    def get_backbone_features(self) -> int:
        """Get RawNet backbone features"""
        return self.rawnet_params['gru_node']
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using RawNet backbone"""
        # x shape: (batch_size, time_samples) for raw audio
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        return self.backbone(x)
    
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Classify using RawNet head"""
        return self.classifier(features)


class RawNetBackbone(nn.Module):
    """RawNet2 backbone for raw waveform processing"""
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
        # First convolution
        self.first_conv = nn.Conv1d(
            params['in_channels'], 
            params['first_conv'], 
            kernel_size=3, 
            padding=1
        )
        self.first_bn = nn.BatchNorm1d(params['first_conv'])
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_ch = params['first_conv']
        
        for i, (filt, nb_blocks) in enumerate(zip(params['filts'], params['blocks'])):
            if isinstance(filt, int):
                out_ch = filt
            else:
                out_ch = filt[-1]
            
            # Add residual blocks
            for j in range(nb_blocks):
                if j == 0 and i > 0:
                    # First block in each stage (except first) uses stride=2
                    self.res_blocks.append(RawNetResBlock(in_ch, out_ch, stride=2))
                else:
                    self.res_blocks.append(RawNetResBlock(in_ch, out_ch, stride=1))
                in_ch = out_ch
        
        # GRU layer for temporal modeling
        self.gru = nn.GRU(
            input_size=in_ch,
            hidden_size=params['gru_node'],
            batch_first=True,
            bidirectional=False
        )
        
        # Fully connected layer
        self.fc = nn.Linear(params['gru_node'], params['nb_fc_node'])
        
    def forward(self, x):
        """Forward pass through RawNet backbone"""
        # x shape: (batch, 1, time)
        
        # First convolution
        x = F.relu(self.first_bn(self.first_conv(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Prepare for GRU: (batch, channels, time) -> (batch, time, channels)
        x = x.transpose(1, 2)
        
        # GRU processing
        x, _ = self.gru(x)
        
        # Take last output
        x = x[:, -1, :]  # (batch, gru_node)
        
        # Final FC layer
        x = F.relu(self.fc(x))
        
        return x


class RawNetResBlock(nn.Module):
    """Residual block for RawNet2"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        return F.relu(out)


# Register audio models with the global registry
def register_audio_models():
    """Register all audio models with the model registry"""
    
    # AASIST
    aasist_config = ModelConfig(
        model_name="aasist",
        input_size=48000,  # 3 seconds at 16kHz
        dropout_rate=0.1,
        model_params={
            'd_args': {
                'nb_samp': 48000,
                'first_conv': 128,
                'filts': [70, [1, 32], [32, 32], [32, 64], [64, 64]],
                'gat_dims': [64, 32],
                'pool_ratios': [0.5, 0.7, 0.5, 0.5]
            }
        }
    )
    model_registry.register_model("aasist", AASIST, aasist_config)
    
    # Wav2Vec2 + AASIST
    wav2vec_aasist_config = ModelConfig(
        model_name="wav2vec_aasist",
        input_size=48000,
        dropout_rate=0.1,
        model_params={
            'wav2vec_model': 'facebook/wav2vec2-base',
            'freeze_wav2vec': True
        }
    )
    model_registry.register_model("wav2vec_aasist", Wav2VecAASIST, wav2vec_aasist_config)
    
    # RawNet2
    rawnet_config = ModelConfig(
        model_name="rawnet2",
        input_size=48000,
        dropout_rate=0.2,
        model_params={
            'rawnet_params': {
                'nb_samp': 48000,
                'first_conv': 128,
                'in_channels': 1,
                'filts': [128, [128, 128], [128, 256], [256, 256]],
                'blocks': [2, 4, 4, 4],
                'nb_fc_node': 1024,
                'gru_node': 1024
            }
        }
    )
    model_registry.register_model("rawnet2", RawNetDeepfake, rawnet_config)
    
    logger.info("Registered 3 audio models: AASIST, Wav2Vec2+AASIST, RawNet2")


# Auto-register models when module is imported
register_audio_models()


def create_aasist_model(input_size: int = 48000,
                       num_classes: int = 2) -> AASIST:
    """Convenience function to create AASIST model"""
    config = ModelConfig(
        model_name="aasist",
        input_size=input_size,
        num_classes=num_classes,
        dropout_rate=0.1
    )
    return AASIST(config)


def create_wav2vec_aasist_model(input_size: int = 48000,
                               num_classes: int = 2,
                               freeze_wav2vec: bool = True) -> Wav2VecAASIST:
    """Convenience function to create Wav2Vec2+AASIST model"""
    config = ModelConfig(
        model_name="wav2vec_aasist",
        input_size=input_size,
        num_classes=num_classes,
        dropout_rate=0.1,
        model_params={
            'freeze_wav2vec': freeze_wav2vec
        }
    )
    return Wav2VecAASIST(config)


def create_rawnet_model(input_size: int = 48000,
                       num_classes: int = 2) -> RawNetDeepfake:
    """Convenience function to create RawNet2 model"""
    config = ModelConfig(
        model_name="rawnet2",
        input_size=input_size,
        num_classes=num_classes,
        dropout_rate=0.2
    )
    return RawNetDeepfake(config)


def main():
    """Demonstrate audio models"""
    print("🎵 AUDIO MODELS DEMO")
    print("=" * 50)
    
    # Test AASIST
    print("\n🔍 Testing AASIST...")
    aasist_model = create_aasist_model()
    
    # Test input (3 seconds at 16kHz)
    test_input = torch.randn(2, 48000)  # Batch of 2 audio samples
    
    with torch.no_grad():
        output = aasist_model(test_input, return_features=True)
        print(f"  ✓ Output shape: {output.logits.shape}")
        print(f"  ✓ Probabilities: {output.probabilities[0].cpu().numpy()}")
        print(f"  ✓ Model params: {sum(p.numel() for p in aasist_model.parameters()):,}")
    
    # Test RawNet2
    print("\n🔍 Testing RawNet2...")
    rawnet_model = create_rawnet_model()
    
    with torch.no_grad():
        output = rawnet_model(test_input, return_features=True)
        print(f"  ✓ Output shape: {output.logits.shape}")
        print(f"  ✓ Probabilities: {output.probabilities[0].cpu().numpy()}")
        print(f"  ✓ Model params: {sum(p.numel() for p in rawnet_model.parameters()):,}")
    
    # Test Wav2Vec2+AASIST (will use CNN fallback if transformers not available)
    print("\n🔍 Testing Wav2Vec2+AASIST...")
    wav2vec_model = create_wav2vec_aasist_model()
    
    with torch.no_grad():
        output = wav2vec_model(test_input, return_features=True)
        print(f"  ✓ Output shape: {output.logits.shape}")
        print(f"  ✓ Probabilities: {output.probabilities[0].cpu().numpy()}")
        print(f"  ✓ Model params: {sum(p.numel() for p in wav2vec_model.parameters()):,}")
    
    # Show model info
    print(f"\n📊 Model Information:")
    aasist_info = aasist_model.get_model_info()
    print(f"  AASIST:")
    print(f"    - Parameters: {aasist_info['total_parameters']:,}")
    print(f"    - Model size: {aasist_info['model_size_mb']:.1f} MB")
    
    rawnet_info = rawnet_model.get_model_info()
    print(f"  RawNet2:")
    print(f"    - Parameters: {rawnet_info['total_parameters']:,}")
    print(f"    - Model size: {rawnet_info['model_size_mb']:.1f} MB")
    
    print(f"\n✅ Audio models ready for training!")


if __name__ == "__main__":
    main()
