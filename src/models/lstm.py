"""
LSTM model for fault detection on time-series sensor data.

Bidirectional LSTM for processing windowed sensor readings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import LSTM_PARAMS, MODELS_DIR, DEVICE


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for time-series fault detection.
    
    Architecture:
        BiLSTM(64) -> BiLSTM(64) -> Dense(64) -> ReLU -> Dropout -> Dense(1) -> Sigmoid
    """
    
    def __init__(
        self,
        input_size: int = LSTM_PARAMS["input_size"],
        hidden_size: int = LSTM_PARAMS["hidden_size"],
        num_layers: int = LSTM_PARAMS["num_layers"],
        bidirectional: bool = LSTM_PARAMS["bidirectional"],
        fc_units: int = LSTM_PARAMS["fc_units"],
        dropout: float = LSTM_PARAMS["dropout"],
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features (sensors)
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            fc_units: Number of units in fully connected layer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * self.num_directions
        self.fc1 = nn.Linear(lstm_output_size, fc_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_units, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, window_size, n_features)
        
        Returns:
            Output tensor of shape (batch,) - fault probability
        """
        # LSTM
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from both directions
        # h_n: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            # Concatenate last hidden states from forward and backward
            h_forward = h_n[-2, :, :]  # Last layer, forward
            h_backward = h_n[-1, :, :]  # Last layer, backward
            hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            hidden = h_n[-1, :, :]  # Last layer hidden state
        
        # Fully connected
        x = F.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x.squeeze(-1)  # (batch,)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head (for transfer learning).
        
        Args:
            x: Input tensor
        
        Returns:
            Feature tensor from fc1 layer
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            hidden = h_n[-1, :, :]
        
        features = F.relu(self.fc1(hidden))
        return features


class LSTMModel:
    """
    Wrapper class for LSTM model with training and evaluation utilities.
    """
    
    def __init__(
        self,
        input_size: int = LSTM_PARAMS["input_size"],
        device: torch.device = DEVICE,
        **kwargs
    ):
        """
        Initialize LSTM model wrapper.
        
        Args:
            input_size: Number of input features
            device: Device to use (cpu, cuda, mps)
            **kwargs: Additional LSTM parameters
        """
        self.device = device
        self.input_size = input_size
        self.model = BiLSTM(input_size=input_size, **kwargs).to(device)
        self.is_fitted = False
        
    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        return self.model
    
    def freeze_base(self) -> None:
        """Freeze LSTM layers for transfer learning."""
        for name, param in self.model.named_parameters():
            if 'lstm' in name:
                param.requires_grad = False
        print("LSTM layers frozen. Only fc layers will be trained.")
    
    def unfreeze_all(self) -> None:
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("All layers unfrozen.")
    
    def save(self, path: Optional[Path] = None, name: str = "lstm") -> Path:
        """Save model state dict."""
        if path is None:
            path = MODELS_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        filepath = path / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
        }, filepath)
        
        print(f"Model saved to: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path, device: torch.device = DEVICE) -> "LSTMModel":
        """Load model from state dict."""
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            input_size=checkpoint['input_size'],
            device=device,
        )
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.is_fitted = True
        
        return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print(f"Device: {DEVICE}")
    
    # Create model
    model = LSTMModel(input_size=21)
    print(f"LSTM parameters: {count_parameters(model.model):,}")
    
    # Test forward pass
    x = torch.randn(32, 30, 21).to(DEVICE)  # (batch, window, features)
    y = model.model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Test feature extraction
    features = model.model.get_features(x)
    print(f"Feature shape: {features.shape}")
