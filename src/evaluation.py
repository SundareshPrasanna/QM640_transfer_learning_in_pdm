"""
Evaluation module for fault detection models.

Provides comprehensive evaluation metrics, statistical tests, and domain shift analysis.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEVICE, MODELS_DIR, ALPHA, SOURCE_DOMAIN, TARGET_DOMAINS
from src.preprocessing import load_processed_data
from src.data_loader import CMAPSSDataset


def evaluate_sklearn_model(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate a scikit-learn model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    return compute_metrics(y, y_pred, y_proba)


def evaluate_torch_model(
    model: torch.nn.Module, 
    test_loader: DataLoader, 
    device: torch.device = DEVICE
) -> Dict[str, float]:
    """Evaluate a PyTorch model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    return compute_metrics(
        np.array(all_labels), 
        np.array(all_preds), 
        np.array(all_probs)
    )


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    else:
        metrics['roc_auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def evaluate_domain_shift(
    model_name: str,
    model,
    source_domain: str = SOURCE_DOMAIN,
    target_domains: List[str] = TARGET_DOMAINS,
    is_torch: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a model on source and all target domains.
    
    Returns DataFrame with metrics for each domain.
    """
    results = []
    
    # Evaluate on source domain
    source_data = load_processed_data(source_domain)
    
    if is_torch:
        dataset = CMAPSSDataset(source_data['X_test'], source_data['y_test'])
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        metrics = evaluate_torch_model(model, loader)
    else:
        metrics = evaluate_sklearn_model(model, source_data['X_test'], source_data['y_test'])
    
    metrics['domain'] = source_domain
    metrics['model'] = model_name
    metrics['domain_type'] = 'source'
    results.append(metrics)
    
    # Evaluate on target domains
    for target in target_domains:
        target_data = load_processed_data(target)
        
        # For neural networks, we need to match input dimensions
        if is_torch:
            # Check if feature dimensions match
            source_features = len(source_data['feature_cols'])
            target_features = len(target_data['feature_cols'])
            
            if source_features != target_features:
                # Need to align features - use only common features or pad
                # For now, we'll skip if dimensions don't match
                print(f"Warning: {target} has {target_features} features, source has {source_features}")
                # We'll create a modified dataset with matching dimensions
                X_test = align_features(
                    target_data['X_test'], 
                    target_data['feature_cols'],
                    source_data['feature_cols']
                )
            else:
                X_test = target_data['X_test']
            
            dataset = CMAPSSDataset(X_test, target_data['y_test'])
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            metrics = evaluate_torch_model(model, loader)
        else:
            # For RF, need to match flattened feature dimensions
            source_flat_dim = source_data['X_train'].shape[1] * source_data['X_train'].shape[2]
            target_flat_dim = target_data['X_test'].shape[1] * target_data['X_test'].shape[2]
            
            if source_flat_dim != target_flat_dim:
                X_test = align_features(
                    target_data['X_test'],
                    target_data['feature_cols'],
                    source_data['feature_cols']
                )
            else:
                X_test = target_data['X_test']
            
            metrics = evaluate_sklearn_model(model, X_test, target_data['y_test'])
        
        metrics['domain'] = target
        metrics['model'] = model_name
        metrics['domain_type'] = 'target'
        results.append(metrics)
    
    return pd.DataFrame(results)


def align_features(
    X: np.ndarray, 
    source_features: List[str], 
    target_features: List[str]
) -> np.ndarray:
    """
    Align features between source and target domains.
    
    Pads missing features with zeros.
    """
    # Create mapping from source features to target features
    source_set = set(source_features)
    target_set = set(target_features)
    
    # Find common features
    common = source_set.intersection(target_set)
    
    # Create new array with target dimensions
    n_samples, window_size, _ = X.shape
    n_target_features = len(target_features)
    
    X_aligned = np.zeros((n_samples, window_size, n_target_features), dtype=X.dtype)
    
    # Copy common features
    for i, feat in enumerate(source_features):
        if feat in target_set:
            target_idx = target_features.index(feat)
            X_aligned[:, :, target_idx] = X[:, :, i]
    
    return X_aligned


def paired_ttest(
    source_metrics: List[float], 
    target_metrics: List[float],
    alpha: float = ALPHA
) -> Dict[str, float]:
    """
    Perform paired t-test for hypothesis testing.
    
    H0: No significant difference between source and target performance
    H1: Significant difference exists
    """
    t_stat, p_value = stats.ttest_rel(source_metrics, target_metrics)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_source: np.ndarray,
    y_pred_target: np.ndarray,
    alpha: float = ALPHA
) -> Dict[str, float]:
    """
    Perform McNemar's test for comparing two classifiers on the same test set.
    """
    # Create contingency table
    correct_source = (y_pred_source == y_true)
    correct_target = (y_pred_target == y_true)
    
    # b: source correct, target wrong
    # c: source wrong, target correct
    b = np.sum(correct_source & ~correct_target)
    c = np.sum(~correct_source & correct_target)
    
    # McNemar's test with continuity correction
    if b + c > 25:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        # Use exact binomial test for small samples
        p_value = stats.binom_test(min(b, c), b + c, 0.5)
    
    return {
        'b': int(b),
        'c': int(c),
        'p_value': p_value,
        'significant': p_value < alpha,
    }


def generate_domain_shift_report(results_df: pd.DataFrame) -> str:
    """Generate a text report of domain shift analysis."""
    report = []
    report.append("="*60)
    report.append("DOMAIN SHIFT ANALYSIS REPORT (RQ1)")
    report.append("="*60)
    
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        
        report.append(f"\n{'-'*40}")
        report.append(f"Model: {model.upper()}")
        report.append(f"{'-'*40}")
        
        source_row = model_df[model_df['domain_type'] == 'source'].iloc[0]
        report.append(f"\nSource Domain ({source_row['domain']}):")
        report.append(f"  Accuracy: {source_row['accuracy']:.4f}")
        report.append(f"  F1-Score: {source_row['f1_score']:.4f}")
        report.append(f"  ROC-AUC: {source_row['roc_auc']:.4f}")
        
        report.append(f"\nTarget Domains (Direct Transfer):")
        target_rows = model_df[model_df['domain_type'] == 'target']
        
        for _, row in target_rows.iterrows():
            degradation = source_row['f1_score'] - row['f1_score']
            report.append(f"\n  {row['domain']}:")
            report.append(f"    Accuracy: {row['accuracy']:.4f}")
            report.append(f"    F1-Score: {row['f1_score']:.4f} (Δ = {degradation:+.4f})")
            report.append(f"    ROC-AUC: {row['roc_auc']:.4f}")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test evaluation
    from src.models.random_forest import RandomForestModel
    
    print("Testing evaluation module...")
    
    # Load RF model
    rf = RandomForestModel.load(MODELS_DIR / "rf_FD002.pkl")
    
    # Evaluate on all domains
    results = evaluate_domain_shift("random_forest", rf, is_torch=False)
    print(results)
