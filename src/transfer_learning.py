"""
Transfer Learning module for fault detection.

Implements fine-tuning strategies for CNN and LSTM models on target domains.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import copy

from src.config import (
    DEVICE, MODELS_DIR, FINE_TUNING_PARAMS, TRAINING_PARAMS,
    SOURCE_DOMAIN, TARGET_DOMAINS, RANDOM_SEED
)
from src.preprocessing import load_processed_data
from src.data_loader import CMAPSSDataset, get_class_weights_tensor
from src.models.cnn import CNNModel, CNN1D
from src.models.lstm import LSTMModel, BiLSTM
from src.training import train_model, evaluate_model, set_seed, EarlyStopping


def align_features_for_transfer(X_target, target_features, source_features):
    """
    Align target domain features to match source domain.
    
    Strategy: Use only features present in both domains, pad missing with zeros.
    """
    n_samples, window_size, _ = X_target.shape
    n_source_features = len(source_features)
    
    # Create aligned array
    X_aligned = np.zeros((n_samples, window_size, n_source_features), dtype=X_target.dtype)
    
    # Map target features to source positions
    for src_idx, src_feat in enumerate(source_features):
        if src_feat in target_features:
            tgt_idx = target_features.index(src_feat)
            X_aligned[:, :, src_idx] = X_target[:, :, tgt_idx]
    
    return X_aligned


def load_pretrained_model(model_type: str, source_domain: str = SOURCE_DOMAIN):
    """
    Load a pretrained model from source domain.
    
    Args:
        model_type: 'cnn' or 'lstm'
        source_domain: Source domain the model was trained on
    
    Returns:
        Tuple of (model, n_features)
    """
    source_data = load_processed_data(source_domain)
    n_features = len(source_data['feature_cols'])
    
    if model_type == 'cnn':
        model_path = MODELS_DIR / f"cnn_{source_domain}.pt"
        model = CNN1D(input_channels=n_features).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'lstm':
        model_path = MODELS_DIR / f"lstm_{source_domain}.pt"
        model = BiLSTM(input_size=n_features).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Loaded pretrained {model_type.upper()} from: {model_path}")
    return model, n_features, source_data['feature_cols']


def freeze_base_layers(model: nn.Module, model_type: str) -> None:
    """
    Freeze base layers of the model (feature extraction layers).
    Only the classification head will be trained.
    """
    if model_type == 'cnn':
        # Freeze conv and batchnorm layers
        for name, param in model.named_parameters():
            if 'conv' in name or 'bn' in name:
                param.requires_grad = False
    elif model_type == 'lstm':
        # Freeze LSTM layers
        for name, param in model.named_parameters():
            if 'lstm' in name:
                param.requires_grad = False
    
    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen base layers: {total - trainable:,} params frozen, {trainable:,} trainable")


def fine_tune_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = FINE_TUNING_PARAMS['epochs'],
    learning_rate: float = FINE_TUNING_PARAMS['learning_rate'],
    freeze_base: bool = FINE_TUNING_PARAMS['freeze_layers'],
    model_type: str = 'cnn',
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Fine-tune a pretrained model on target domain data.
    
    Args:
        model: Pretrained model
        train_loader: Target domain training data
        val_loader: Target domain validation data
        epochs: Number of fine-tuning epochs
        learning_rate: Learning rate (typically lower than initial training)
        freeze_base: Whether to freeze base layers
        model_type: 'cnn' or 'lstm'
        verbose: Whether to print progress
    
    Returns:
        Training history dictionary
    """
    set_seed()
    
    # Optionally freeze base layers
    if freeze_base:
        freeze_base_layers(model, model_type)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, mode='min')
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    if verbose:
        print(f"\nFine-tuning on target domain...")
        print(f"  Epochs: {epochs}, LR: {learning_rate}, Freeze base: {freeze_base}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.float().to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)
        
        train_loss = total_loss / total_samples
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.float().to(DEVICE)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss = val_loss / len(val_labels)
        
        # Compute F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
        
        # Early stopping
        if early_stopping(val_loss):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def prepare_target_data(
    target_domain: str,
    source_features: List[str],
    label_fraction: float = 1.0,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare target domain data for fine-tuning.
    
    Args:
        target_domain: Target domain name
        source_features: Feature columns from source domain
        label_fraction: Fraction of labeled data to use
        val_split: Validation split ratio
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    target_data = load_processed_data(target_domain)
    target_features = target_data['feature_cols']
    
    # Align features
    X_train = target_data['X_train']
    y_train = target_data['y_train']
    X_test = target_data['X_test']
    y_test = target_data['y_test']
    
    if len(target_features) != len(source_features):
        X_train = align_features_for_transfer(X_train, target_features, source_features)
        X_test = align_features_for_transfer(X_test, target_features, source_features)
        print(f"  Aligned features: {len(target_features)} -> {len(source_features)}")
    
    # Sample subset if label_fraction < 1.0
    if label_fraction < 1.0:
        n_samples = int(len(y_train) * label_fraction)
        np.random.seed(RANDOM_SEED)
        indices = np.random.choice(len(y_train), size=n_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        print(f"  Using {label_fraction*100:.0f}% of labels: {n_samples} samples")
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_split,
        stratify=y_train,
        random_state=RANDOM_SEED
    )
    
    # Create datasets
    train_dataset = CMAPSSDataset(X_train, y_train)
    val_dataset = CMAPSSDataset(X_val, y_val)
    test_dataset = CMAPSSDataset(X_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader


def run_fine_tuning_experiment(
    model_type: str,
    target_domain: str,
    label_fraction: float = 0.2,
    freeze_base: bool = True,
) -> Dict:
    """
    Run a complete fine-tuning experiment.
    
    Args:
        model_type: 'cnn' or 'lstm'
        target_domain: Target domain to fine-tune on
        label_fraction: Fraction of target labels to use
        freeze_base: Whether to freeze base layers
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*50}")
    print(f"Fine-tuning {model_type.upper()} on {target_domain}")
    print(f"{'='*50}")
    
    # Load pretrained model
    model, n_features, source_features = load_pretrained_model(model_type)
    
    # Prepare target data
    train_loader, val_loader, test_loader = prepare_target_data(
        target_domain, source_features, label_fraction
    )
    
    # Fine-tune
    history = fine_tune_model(
        model,
        train_loader,
        val_loader,
        freeze_base=freeze_base,
        model_type=model_type,
    )
    
    # Evaluate
    from src.evaluation import evaluate_torch_model
    test_metrics = evaluate_torch_model(model, test_loader)
    
    print(f"\nTest Metrics after fine-tuning:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    return {
        'model_type': model_type,
        'target_domain': target_domain,
        'label_fraction': label_fraction,
        'freeze_base': freeze_base,
        'history': history,
        'test_metrics': test_metrics,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing transfer learning module...")
    
    result = run_fine_tuning_experiment(
        model_type='cnn',
        target_domain='FD001',
        label_fraction=0.2,
        freeze_base=True,
    )
    
    print(f"\nExperiment complete!")
    print(f"Final F1: {result['test_metrics']['f1_score']:.4f}")
