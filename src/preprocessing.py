"""
Data preprocessing module for C-MAPSS turbofan engine dataset.

This module handles:
1. Loading raw C-MAPSS data files
2. Computing RUL (Remaining Useful Life) for training data
3. Creating binary fault labels (RUL <= threshold)
4. Normalizing sensor readings
5. Creating sliding windows for CNN/LSTM models
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import StandardScaler
import pickle

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LABELS_DIR,
    COLUMN_NAMES,
    DATASETS,
    FAULT_THRESHOLD,
    WINDOW_SIZE,
    SENSORS_TO_USE,
    OP_SETTINGS,
)


def load_raw_data(dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Load raw C-MAPSS data files for a given dataset.
    
    Args:
        dataset_name: One of 'FD001', 'FD002', 'FD003', 'FD004'
    
    Returns:
        Tuple of (train_df, test_df, test_rul)
    """
    files = DATASETS[dataset_name]
    
    # Load training data
    train_path = RAW_DATA_DIR / files["train"]
    train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    
    # Load test data
    test_path = RAW_DATA_DIR / files["test"]
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMN_NAMES)
    
    # Load RUL values for test data
    rul_path = RAW_DATA_DIR / files["rul"]
    test_rul = pd.read_csv(rul_path, sep=r'\s+', header=None).values.flatten()
    
    print(f"Loaded {dataset_name}:")
    print(f"  Train: {len(train_df)} rows, {train_df['unit_id'].nunique()} engines")
    print(f"  Test: {len(test_df)} rows, {test_df['unit_id'].nunique()} engines")
    
    return train_df, test_df, test_rul


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for each row in training data.
    
    RUL = max_cycle_for_unit - current_cycle
    
    Args:
        df: DataFrame with 'unit_id' and 'cycle' columns
    
    Returns:
        DataFrame with added 'RUL' column
    """
    df = df.copy()
    
    # Get max cycle for each unit (end of life)
    max_cycles = df.groupby('unit_id')['cycle'].max().to_dict()
    
    # Compute RUL: max_cycle - current_cycle
    df['RUL'] = df.apply(lambda row: max_cycles[row['unit_id']] - row['cycle'], axis=1)
    
    return df


def compute_test_rul(df: pd.DataFrame, test_rul: np.ndarray) -> pd.DataFrame:
    """
    Compute RUL for test data using the provided RUL values.
    
    The test_rul array contains the RUL at the LAST cycle of each engine.
    We compute RUL for all cycles by adding the offset.
    
    Args:
        df: Test DataFrame
        test_rul: Array of RUL values at end of each engine's test sequence
    
    Returns:
        DataFrame with added 'RUL' column
    """
    df = df.copy()
    
    # Get max cycle for each unit in test data
    max_cycles = df.groupby('unit_id')['cycle'].max().to_dict()
    
    # Create RUL mapping: unit_id -> RUL at last cycle
    unit_ids = sorted(df['unit_id'].unique())
    rul_at_end = {uid: test_rul[i] for i, uid in enumerate(unit_ids)}
    
    # Compute RUL: (max_cycle - current_cycle) + RUL_at_end
    df['RUL'] = df.apply(
        lambda row: (max_cycles[row['unit_id']] - row['cycle']) + rul_at_end[row['unit_id']], 
        axis=1
    )
    
    return df


def create_fault_labels(df: pd.DataFrame, threshold: int = FAULT_THRESHOLD) -> pd.DataFrame:
    """
    Create binary fault labels based on RUL threshold.
    
    Args:
        df: DataFrame with 'RUL' column
        threshold: RUL threshold for fault classification
    
    Returns:
        DataFrame with added 'fault_label' column (1 = fault, 0 = healthy)
    """
    df = df.copy()
    df['fault_label'] = (df['RUL'] <= threshold).astype(int)
    
    # Print class distribution
    healthy = (df['fault_label'] == 0).sum()
    fault = (df['fault_label'] == 1).sum()
    print(f"  Labels (threshold={threshold}): Healthy={healthy} ({100*healthy/len(df):.1f}%), Fault={fault} ({100*fault/len(df):.1f}%)")
    
    return df


def identify_constant_sensors(df: pd.DataFrame, std_threshold: float = 0.001) -> List[str]:
    """
    Identify sensors with near-constant values (very low standard deviation).
    
    Args:
        df: DataFrame with sensor columns
        std_threshold: Absolute standard deviation threshold below which sensor is considered constant
    
    Returns:
        List of sensor names to drop
    """
    sensors = [col for col in df.columns if col.startswith('sensor_')]
    constant_sensors = []
    
    for sensor in sensors:
        std = df[sensor].std()
        
        # Consider constant if std is essentially zero
        if std < std_threshold:
            constant_sensors.append(sensor)
    
    return constant_sensors


def normalize_data(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normalize features using StandardScaler fitted on training data.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of columns to normalize
    
    Returns:
        Tuple of (normalized_train_df, normalized_test_df, scaler)
    """
    scaler = StandardScaler()
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Fit on training data
    scaler.fit(train_df[feature_cols])
    
    # Transform both
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, test_df, scaler


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = WINDOW_SIZE,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows for time-series models (CNN/LSTM).
    
    For each engine, we create overlapping windows of sensor readings.
    The label for each window is the fault_label at the END of the window.
    
    Args:
        df: DataFrame with sensor readings and fault_label
        feature_cols: Columns to include in windows
        window_size: Number of time steps in each window
        step: Step size between windows
    
    Returns:
        Tuple of (X, y, unit_ids) where:
            X: shape (n_samples, window_size, n_features)
            y: shape (n_samples,)
            unit_ids: shape (n_samples,) - engine ID for each sample
    """
    X_list = []
    y_list = []
    unit_list = []
    
    for unit_id, group in df.groupby('unit_id'):
        # Sort by cycle
        group = group.sort_values('cycle')
        
        # Get feature values and labels
        features = group[feature_cols].values
        labels = group['fault_label'].values
        
        # Skip if not enough data points
        if len(group) < window_size:
            continue
        
        # Create windows
        for i in range(0, len(group) - window_size + 1, step):
            X_list.append(features[i:i + window_size])
            y_list.append(labels[i + window_size - 1])  # Label at end of window
            unit_list.append(unit_id)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    unit_ids = np.array(unit_list)
    
    return X, y, unit_ids


def preprocess_dataset(
    dataset_name: str,
    save: bool = True
) -> Dict:
    """
    Full preprocessing pipeline for a C-MAPSS dataset.
    
    Args:
        dataset_name: One of 'FD001', 'FD002', 'FD003', 'FD004'
        save: Whether to save processed data to disk
    
    Returns:
        Dictionary with processed data
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name}")
    print('='*60)
    
    # Load raw data
    train_df, test_df, test_rul = load_raw_data(dataset_name)
    
    # Compute RUL
    train_df = compute_rul(train_df)
    test_df = compute_test_rul(test_df, test_rul)
    
    # Create fault labels
    print("\nTraining data:")
    train_df = create_fault_labels(train_df)
    print("Test data:")
    test_df = create_fault_labels(test_df)
    
    # Identify constant sensors from training data
    constant_sensors = identify_constant_sensors(train_df)
    print(f"\nDropping constant sensors: {constant_sensors}")
    
    # Define feature columns (all sensors except constant ones)
    feature_cols = [s for s in SENSORS_TO_USE if s not in constant_sensors]
    print(f"Using {len(feature_cols)} sensors: {feature_cols}")
    
    # Normalize
    train_df, test_df, scaler = normalize_data(train_df, test_df, feature_cols)
    
    # Create sliding windows
    print(f"\nCreating sliding windows (size={WINDOW_SIZE})...")
    X_train, y_train, units_train = create_sliding_windows(train_df, feature_cols)
    X_test, y_test, units_test = create_sliding_windows(test_df, feature_cols)
    
    print(f"  Train: {X_train.shape[0]} samples, shape {X_train.shape}")
    print(f"  Test: {X_test.shape[0]} samples, shape {X_test.shape}")
    
    # Class distribution in windowed data
    print(f"\nWindowed class distribution:")
    print(f"  Train: Healthy={np.sum(y_train==0)}, Fault={np.sum(y_train==1)}")
    print(f"  Test: Healthy={np.sum(y_test==0)}, Fault={np.sum(y_test==1)}")
    
    result = {
        'dataset_name': dataset_name,
        'X_train': X_train,
        'y_train': y_train,
        'units_train': units_train,
        'X_test': X_test,
        'y_test': y_test,
        'units_test': units_test,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'train_df': train_df,
        'test_df': test_df,
    }
    
    if save:
        save_processed_data(result)
    
    return result


def save_processed_data(data: Dict) -> None:
    """Save processed data to disk."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    dataset_name = data['dataset_name']
    output_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nSaved processed data to: {output_path}")


def load_processed_data(dataset_name: str) -> Dict:
    """Load processed data from disk."""
    input_path = PROCESSED_DATA_DIR / f"{dataset_name}_processed.pkl"
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    return data


def preprocess_all_datasets() -> None:
    """Preprocess all C-MAPSS datasets."""
    for dataset_name in DATASETS.keys():
        preprocess_dataset(dataset_name)
    
    print("\n" + "="*60)
    print("All datasets processed successfully!")
    print("="*60)


if __name__ == "__main__":
    preprocess_all_datasets()
