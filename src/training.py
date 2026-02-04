"""
Training module for fault detection models.

Provides unified training interface for CNN and LSTM models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import TRAINING_PARAMS, DEVICE, RANDOM_SEED
from src.models.cnn import CNNModel
from src.models.lstm import LSTMModel


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy/f1
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().to(device)
        
        optimizer.zero_grad()
        
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Validate model.
    
    Returns:
        Tuple of (average loss, accuracy, metrics dict)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.float().to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    avg_loss = total_loss / len(all_labels)
    accuracy = (all_preds == all_labels).mean()
    
    # Compute additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }
    
    return avg_loss, accuracy, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = TRAINING_PARAMS["epochs"],
    learning_rate: float = TRAINING_PARAMS["learning_rate"],
    weight_decay: float = TRAINING_PARAMS["weight_decay"],
    class_weights: Optional[torch.Tensor] = None,
    patience: int = TRAINING_PARAMS["early_stopping_patience"],
    device: torch.device = DEVICE,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Train a neural network model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: L2 regularization
        class_weights: Optional weights for imbalanced classes
        patience: Early stopping patience
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history
    """
    set_seed()
    
    # Loss function with class weights
    if class_weights is not None:
        # For binary classification, use pos_weight for BCEWithLogitsLoss
        pos_weight = class_weights[1] / class_weights[0]
        criterion = nn.BCELoss()  # We use BCELoss since model outputs sigmoid
    else:
        criterion = nn.BCELoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    if verbose:
        print(f"\nTraining on {device}...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_metrics = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_metrics['f1_score'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_metrics['f1_score']:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {elapsed:.1f}s")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Evaluate model on test data.
    
    Returns:
        Dictionary of evaluation metrics
    """
    criterion = nn.BCELoss()
    _, _, metrics = validate(model, test_loader, criterion, device)
    return metrics


if __name__ == "__main__":
    from src.data_loader import get_data_loaders, get_class_weights_tensor
    from src.models.cnn import CNNModel, count_parameters
    
    print("Testing training module...")
    
    # Get data
    train_loader, val_loader, test_loader = get_data_loaders("FD002", batch_size=64)
    class_weights = get_class_weights_tensor("FD002")
    
    # Create and train CNN for a few epochs (quick test)
    model = CNNModel(input_channels=21)
    print(f"CNN parameters: {count_parameters(model.model):,}")
    
    # Train for just 3 epochs as test
    history = train_model(
        model.model,
        train_loader,
        val_loader,
        epochs=3,
        class_weights=class_weights,
        verbose=True
    )
    
    # Evaluate
    metrics = evaluate_model(model.model, test_loader)
    print("\nTest Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
