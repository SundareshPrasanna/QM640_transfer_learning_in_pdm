"""
Train baseline models on source domain (FD002).

Trains Random Forest, CNN, and LSTM on FD002 and saves models.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from datetime import datetime

from src.config import (
    DEVICE, TRAINING_PARAMS, MODELS_DIR, SOURCE_DOMAIN, RANDOM_SEED
)
from src.preprocessing import load_processed_data
from src.data_loader import get_data_loaders, get_class_weights_tensor
from src.models.random_forest import RandomForestModel
from src.models.cnn import CNNModel, count_parameters
from src.models.lstm import LSTMModel
from src.training import train_model, evaluate_model, set_seed


def train_random_forest(dataset_name: str = SOURCE_DOMAIN) -> dict:
    """Train Random Forest on source domain."""
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    # Load data
    data = load_processed_data(dataset_name)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Train
    model = RandomForestModel()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    print(f"\nTraining Metrics:")
    for name, value in train_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\nTest Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save
    model.save(name=f"rf_{dataset_name}")
    
    return {
        'model': model,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
    }


def train_cnn(dataset_name: str = SOURCE_DOMAIN) -> dict:
    """Train CNN on source domain."""
    print("\n" + "="*60)
    print("TRAINING CNN")
    print("="*60)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name, batch_size=TRAINING_PARAMS['batch_size']
    )
    class_weights = get_class_weights_tensor(dataset_name)
    
    # Get number of features
    data = load_processed_data(dataset_name)
    n_features = len(data['feature_cols'])
    
    # Create model
    set_seed()
    model = CNNModel(input_channels=n_features)
    print(f"CNN parameters: {count_parameters(model.model):,}")
    
    # Train
    history = train_model(
        model.model,
        train_loader,
        val_loader,
        epochs=TRAINING_PARAMS['epochs'],
        class_weights=class_weights,
        verbose=True
    )
    
    # Evaluate
    test_metrics = evaluate_model(model.model, test_loader)
    
    print(f"\nTest Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save
    model.save(name=f"cnn_{dataset_name}")
    
    return {
        'model': model,
        'history': history,
        'test_metrics': test_metrics,
    }


def train_lstm(dataset_name: str = SOURCE_DOMAIN) -> dict:
    """Train LSTM on source domain."""
    print("\n" + "="*60)
    print("TRAINING LSTM")
    print("="*60)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name, batch_size=TRAINING_PARAMS['batch_size']
    )
    class_weights = get_class_weights_tensor(dataset_name)
    
    # Get number of features
    data = load_processed_data(dataset_name)
    n_features = len(data['feature_cols'])
    
    # Create model
    set_seed()
    model = LSTMModel(input_size=n_features)
    print(f"LSTM parameters: {count_parameters(model.model):,}")
    
    # Train
    history = train_model(
        model.model,
        train_loader,
        val_loader,
        epochs=TRAINING_PARAMS['epochs'],
        class_weights=class_weights,
        verbose=True
    )
    
    # Evaluate
    test_metrics = evaluate_model(model.model, test_loader)
    
    print(f"\nTest Metrics:")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Save
    model.save(name=f"lstm_{dataset_name}")
    
    return {
        'model': model,
        'history': history,
        'test_metrics': test_metrics,
    }


def train_all_baselines():
    """Train all baseline models on source domain."""
    print("="*60)
    print("BASELINE MODEL TRAINING")
    print(f"Source Domain: {SOURCE_DOMAIN}")
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # Train each model
    results['random_forest'] = train_random_forest()
    results['cnn'] = train_cnn()
    results['lstm'] = train_lstm()
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\n{'Model':<15} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-"*50)
    
    for model_name, result in results.items():
        metrics = result['test_metrics']
        print(f"{model_name:<15} {metrics['accuracy']:>10.4f} {metrics['f1_score']:>10.4f} {metrics['roc_auc']:>10.4f}")
    
    print("\n" + "="*60)
    print("All baseline models trained and saved successfully!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = train_all_baselines()
