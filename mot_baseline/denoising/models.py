"""Phase 6A: Denoising Models

Two architectures:
1. CausalDilatedCNN — WaveNet-like, only uses past frames (causal)
2. BiLSTMDenoiser — uses past + future frames (non-causal, for offline/retrospective correction)

Both take input (batch, window_size, 2) and output (batch, 1) — the denoised range at the target frame.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D convolution with causal padding (only looks at past)."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             dilation=dilation)
    
    def forward(self, x):
        # x: (batch, channels, time)
        x = F.pad(x, (self.padding, 0))  # left pad only (causal)
        return self.conv(x)


class CausalDilatedCNN(nn.Module):
    """WaveNet-like 1D causal dilated CNN for FTM denoising.
    
    Architecture:
        Input (batch, window, 2) → Conv layers with exponentially increasing dilation
        → Global pooling over last position → Linear → denoised range
    
    Receptive field with 4 layers, kernel=3, dilation=[1,2,4,8]: 
        1 + (3-1)*1 + (3-1)*2 + (3-1)*4 + (3-1)*8 = 1+2+4+8+16 = 31 frames = 3.1s
    """
    
    def __init__(self, in_features: int = 2, hidden_dim: int = 32, 
                 n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Conv1d(in_features, hidden_dim, 1)
        
        # Dilated causal conv layers with residual connections
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.layers.append(
                CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output head: take the last time step and project to scalar
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, window_size, in_features)
        Returns:
            (batch, 1) — denoised range at last time step
        """
        # (batch, window, features) → (batch, features, window)
        x = x.transpose(1, 2)
        
        # Input projection
        h = self.input_proj(x)  # (batch, hidden, window)
        
        # Dilated causal conv stack with residual
        for conv, norm in zip(self.layers, self.norms):
            residual = h
            h = conv(h)  # (batch, hidden, window)
            h = h.transpose(1, 2)  # (batch, window, hidden)
            h = norm(h)
            h = h.transpose(1, 2)  # (batch, hidden, window)
            h = F.gelu(h)
            h = h + residual  # residual connection
        
        # Take the last time step (causal: prediction at t=last)
        h_last = h[:, :, -1]  # (batch, hidden)
        
        # Output projection
        out = self.output_head(h_last)  # (batch, 1)
        return out


class BiLSTMDenoiser(nn.Module):
    """Bi-directional LSTM for FTM denoising (non-causal).
    
    Uses both past and future context. For offline/retrospective correction.
    
    Architecture:
        Input (batch, window, 2) → BiLSTM → take center hidden state → Linear → denoised range
    """
    
    def __init__(self, in_features: int = 2, hidden_dim: int = 32, 
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        
        # Output head: 2*hidden_dim (bidirectional) → 1
        self.output_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, window_size, in_features)
        Returns:
            (batch, 1) — denoised range at center time step
        """
        # BiLSTM over the full window
        output, _ = self.lstm(x)  # (batch, window, 2*hidden)
        
        # Take the center time step (non-causal: uses both past and future)
        center_idx = x.shape[1] // 2
        h_center = output[:, center_idx, :]  # (batch, 2*hidden)
        
        # Output projection
        out = self.output_head(h_center)  # (batch, 1)
        return out


class CausalLSTMDenoiser(nn.Module):
    """Unidirectional LSTM for FTM denoising (causal).
    
    Only uses past context. For real-time MOT.
    """
    
    def __init__(self, in_features: int = 2, hidden_dim: int = 32, 
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, window_size, in_features)
        Returns:
            (batch, 1) — denoised range at last time step (causal)
        """
        output, _ = self.lstm(x)  # (batch, window, hidden)
        h_last = output[:, -1, :]  # (batch, hidden)
        out = self.output_head(h_last)  # (batch, 1)
        return out


# ===========================================================================
# Model factory
# ===========================================================================
MODEL_REGISTRY = {
    "causal_cnn": CausalDilatedCNN,
    "bilstm": BiLSTMDenoiser,
    "causal_lstm": CausalLSTMDenoiser,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Model parameter counts:")
    for name in MODEL_REGISTRY:
        model = build_model(name)
        n = count_params(model)
        print(f"  {name}: {n:,} params")
    
    # Test forward pass
    print("\nForward pass test (batch=4, window=15, features=2):")
    x = torch.randn(4, 15, 2)
    for name in MODEL_REGISTRY:
        model = build_model(name)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"  {name}: input={x.shape} → output={out.shape}")
