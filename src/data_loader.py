"""
PyTorch data loaders for C-MAPSS turbofan engine dataset.

Provides DataLoader classes for training, validation, and transfer learning experiments.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from typing import Tuple, Dict, Optional, List
from sklearn.model_selection import train_test_split

from src.config import (
    TRAINING_PARAMS,
    RANDOM_SEED,
    DEVICE,
)
from src.preprocessing import load_processed_data


class CMAPSSDataset(Dataset):
    """
    PyTorch Dataset for C-MAPSS turbofan engine data.
    
    Handles windowed time-series data for fault detection.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        unit_ids: Optional[np.ndarray] = None
    ):
        """
        Args:
            X: Feature array of shape (n_samples, window_size, n_features)
            y: Label array of shape (n_samples,)
            unit_ids: Optional engine unit IDs for each sample
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.unit_ids = unit_ids
        
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        class_counts = torch.bincount(self.y)
        total = len(self.y)
        weights = total / (len(class_counts) * class_counts.float())
        return weights
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.y]
        return sample_weights


def get_data_loaders(
    dataset_name: str,
    batch_size: int = TRAINING_PARAMS["batch_size"],
    val_split: float = 0.2,
    weighted_sampling: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test data loaders for a dataset.
    
    Args:
        dataset_name: One of 'FD001', 'FD002', 'FD003', 'FD004'
        batch_size: Batch size for training
        val_split: Fraction of training data to use for validation
        weighted_sampling: Whether to use weighted sampling for class imbalance
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load processed data
    data = load_processed_data(dataset_name)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Split training data into train and validation
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
    
    # Create sampler for handling class imbalance
    if weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Sampler handles randomization
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type != "cpu" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type != "cpu" else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type != "cpu" else False,
    )
    
    print(f"Data loaders created for {dataset_name}:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def get_transfer_data_loaders(
    source_name: str,
    target_name: str,
    target_label_fraction: float = 1.0,
    batch_size: int = TRAINING_PARAMS["batch_size"],
) -> Dict[str, DataLoader]:
    """
    Get data loaders for transfer learning experiments.
    
    Args:
        source_name: Source domain dataset (e.g., 'FD002')
        target_name: Target domain dataset (e.g., 'FD001')
        target_label_fraction: Fraction of target training data to use (for label efficiency experiments)
        batch_size: Batch size
    
    Returns:
        Dictionary with source and target data loaders
    """
    # Source domain loaders
    source_train_loader, source_val_loader, _ = get_data_loaders(
        source_name, batch_size=batch_size
    )
    
    # Target domain - load data separately to control labeled fraction
    target_data = load_processed_data(target_name)
    
    X_target = target_data['X_train']
    y_target = target_data['y_train']
    
    # Sample subset if fraction < 1.0
    if target_label_fraction < 1.0:
        n_samples = int(len(y_target) * target_label_fraction)
        indices = np.random.RandomState(RANDOM_SEED).choice(
            len(y_target), size=n_samples, replace=False
        )
        X_target = X_target[indices]
        y_target = y_target[indices]
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_target, y_target,
        test_size=0.2,
        stratify=y_target,
        random_state=RANDOM_SEED
    )
    
    target_train_dataset = CMAPSSDataset(X_train, y_train)
    target_val_dataset = CMAPSSDataset(X_val, y_val)
    target_test_dataset = CMAPSSDataset(
        target_data['X_test'], target_data['y_test']
    )
    
    # Weighted sampling for target training
    sample_weights = target_train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=batch_size,
        sampler=sampler,
    )
    
    target_val_loader = DataLoader(
        target_val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    print(f"\nTransfer learning setup: {source_name} → {target_name}")
    print(f"  Target label fraction: {target_label_fraction*100:.0f}%")
    print(f"  Target train samples: {len(target_train_dataset)}")
    
    return {
        'source_train': source_train_loader,
        'source_val': source_val_loader,
        'target_train': target_train_loader,
        'target_val': target_val_loader,
        'target_test': target_test_loader,
    }


def get_class_weights_tensor(dataset_name: str) -> torch.Tensor:
    """Get class weights tensor for loss function."""
    data = load_processed_data(dataset_name)
    y = data['y_train']
    
    class_counts = np.bincount(y)
    total = len(y)
    weights = total / (len(class_counts) * class_counts.astype(float))
    
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


if __name__ == "__main__":
    # Test data loaders
    print("Testing data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders("FD002")
    
    # Get a batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nBatch shapes: X={X_batch.shape}, y={y_batch.shape}")
    
    # Test transfer loaders
    print("\n" + "="*50)
    loaders = get_transfer_data_loaders("FD002", "FD001", target_label_fraction=0.1)
