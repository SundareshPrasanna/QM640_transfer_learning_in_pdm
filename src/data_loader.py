"""
PyTorch data loaders for C-MAPSS turbofan engine dataset.

Provides DataLoader classes for training, validation, and transfer learning experiments.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
        class_counts = torch.bincount(self.y, minlength=2).float().clamp_min(1.0)
        total = len(self.y)
        weights = total / (2 * class_counts)
        return weights
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.y]
        return sample_weights


def split_by_engine(
    X: np.ndarray,
    y: np.ndarray,
    unit_ids: np.ndarray,
    val_split: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data at engine level to avoid leakage across overlapping windows.
    """
    unique_units = np.unique(unit_ids)

    # If too few engines are available (e.g., extreme label-efficiency subsets),
    # fall back to a window-level split to keep the experiment runnable.
    if len(unique_units) < 2:
        indices = np.arange(len(y))
        stratify_labels = y if len(np.unique(y)) > 1 else None
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels,
        )
        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    train_units, val_units = train_test_split(
        unique_units,
        test_size=val_split,
        random_state=random_state,
        shuffle=True,
    )

    train_mask = np.isin(unit_ids, train_units)
    val_mask = np.isin(unit_ids, val_units)

    return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


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
    units_train = data.get('units_train')
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Split training data into train and validation at engine level
    if units_train is not None:
        X_train, X_val, y_train, y_val = split_by_engine(
            X_train,
            y_train,
            units_train,
            val_split=val_split,
            random_state=RANDOM_SEED,
        )
    else:
        # Backward-compatible fallback if unit IDs are unavailable
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_split,
            stratify=y_train,
            random_state=RANDOM_SEED,
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
    units_target = target_data.get('units_train')
    
    # Sample subset if fraction < 1.0 (engine-level sampling to avoid leakage)
    if target_label_fraction < 1.0 and units_target is not None:
        unique_units = np.unique(units_target)
        n_units = max(1, int(np.ceil(len(unique_units) * target_label_fraction)))
        selected_units = np.random.RandomState(RANDOM_SEED).choice(
            unique_units, size=n_units, replace=False
        )
        subset_mask = np.isin(units_target, selected_units)
        X_target = X_target[subset_mask]
        y_target = y_target[subset_mask]
        units_target = units_target[subset_mask]
    
    # Split into train/val
    if units_target is not None:
        X_train, X_val, y_train, y_val = split_by_engine(
            X_target,
            y_target,
            units_target,
            val_split=0.2,
            random_state=RANDOM_SEED,
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_target,
            y_target,
            test_size=0.2,
            stratify=y_target,
            random_state=RANDOM_SEED,
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
    
    class_counts = np.bincount(y, minlength=2).astype(float)
    class_counts = np.clip(class_counts, 1.0, None)
    total = len(y)
    weights = total / (2 * class_counts)
    
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
