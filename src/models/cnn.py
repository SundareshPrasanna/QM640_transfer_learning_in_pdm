"""
CNN model for fault detection on time-series sensor data.

1D Convolutional Neural Network for processing windowed sensor readings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CNN_PARAMS, MODELS_DIR, DEVICE


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for time-series fault detection.
    
    Architecture:
        Conv1D(64) -> ReLU -> BatchNorm ->
        Conv1D(128) -> ReLU -> BatchNorm -> MaxPool ->
        Conv1D(128) -> ReLU -> BatchNorm -> MaxPool ->
        GlobalAvgPool -> Dense(64) -> ReLU -> Dropout -> Dense(1) -> Sigmoid
    """
    
    def __init__(
        self,
        input_channels: int = CNN_PARAMS["input_channels"],
        conv_filters: list = CNN_PARAMS["conv_filters"],
        kernel_size: int = CNN_PARAMS["kernel_size"],
        fc_units: int = CNN_PARAMS["fc_units"],
        dropout: float = CNN_PARAMS["dropout"],
    ):
        """
        Initialize CNN model.
        
        Args:
            input_channels: Number of input features (sensors)
            conv_filters: List of filter sizes for conv layers
            kernel_size: Kernel size for conv layers
            fc_units: Number of units in fully connected layer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_channels = input_channels
        
        # Convolutional layers
        # Input: (batch, window_size, n_features) -> transpose to (batch, n_features, window_size)
        self.conv1 = nn.Conv1d(input_channels, conv_filters[0], kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(conv_filters[0])
        
        self.conv2 = nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(conv_filters[1])
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(conv_filters[1], conv_filters[2], kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(conv_filters[2])
        self.pool2 = nn.MaxPool1d(2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_filters[2], fc_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_units, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, window_size, n_features)
        
        Returns:
            Output tensor of shape (batch, 1) - fault probability
        """
        # Transpose: (batch, window_size, features) -> (batch, features, window_size)
        x = x.transpose(1, 2)
        
        # Conv blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        
        # Global pooling
        x = self.global_pool(x)  # (batch, filters, 1)
        x = x.squeeze(-1)  # (batch, filters)
        
        # Fully connected
        x = F.relu(self.fc1(x))
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
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = F.relu(self.fc1(x))
        return x


class CNNModel:
    """
    Wrapper class for CNN model with training and evaluation utilities.
    """
    
    def __init__(
        self,
        input_channels: int = CNN_PARAMS["input_channels"],
        device: torch.device = DEVICE,
        **kwargs
    ):
        """
        Initialize CNN model wrapper.
        
        Args:
            input_channels: Number of input features
            device: Device to use (cpu, cuda, mps)
            **kwargs: Additional CNN parameters
        """
        self.device = device
        self.input_channels = input_channels
        self.model = CNN1D(input_channels=input_channels, **kwargs).to(device)
        self.is_fitted = False
        
    def get_model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        return self.model
    
    def freeze_base(self) -> None:
        """Freeze convolutional layers for transfer learning."""
        for name, param in self.model.named_parameters():
            if 'conv' in name or 'bn' in name:
                param.requires_grad = False
        print("Base layers frozen. Only fc layers will be trained.")
    
    def unfreeze_all(self) -> None:
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("All layers unfrozen.")
    
    def save(self, path: Optional[Path] = None, name: str = "cnn") -> Path:
        """Save model state dict."""
        if path is None:
            path = MODELS_DIR
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        filepath = path / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_channels': self.input_channels,
        }, filepath)
        
        print(f"Model saved to: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path, device: torch.device = DEVICE) -> "CNNModel":
        """Load model from state dict."""
        checkpoint = torch.load(filepath, map_location=device)
        
        model = cls(
            input_channels=checkpoint['input_channels'],
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
    model = CNNModel(input_channels=21)
    print(f"CNN parameters: {count_parameters(model.model):,}")
    
    # Test forward pass
    x = torch.randn(32, 30, 21).to(DEVICE)  # (batch, window, features)
    y = model.model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Test feature extraction
    features = model.model.get_features(x)
    print(f"Feature shape: {features.shape}")
