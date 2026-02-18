"""
Random Forest model for fault detection.

Flattens time-series windows into feature vectors for scikit-learn RF.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from typing import Dict, Any, Optional
import pickle
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import RF_PARAMS, MODELS_DIR, RANDOM_SEED


class RandomForestModel:
    """
    Random Forest model wrapper for fault detection.
    
    Converts each (window_size, n_features) window into statistical features.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Random Forest model.
        
        Args:
            params: Optional dict of RF hyperparameters (overrides defaults)
        """
        self.params = RF_PARAMS.copy()
        if params:
            self.params.update(params)
        
        self.model = RandomForestClassifier(**self.params)
        self.is_fitted = False
        self.n_features_in_ = None
        
    def _extract_window_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from 3D window data for RF.

        Args:
            X: Array of shape (n_samples, window_size, n_features)

        Returns:
            Array of shape (n_samples, n_features * n_stats)
        """
        # Basic summary stats per sensor
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        min_v = X.min(axis=1)
        max_v = X.max(axis=1)
        p25 = np.percentile(X, 25, axis=1)
        p50 = np.percentile(X, 50, axis=1)
        p75 = np.percentile(X, 75, axis=1)

        # Linear trend (slope) per sensor across window time
        t = np.arange(X.shape[1], dtype=np.float32)
        t_centered = t - t.mean()
        denom = np.sum(t_centered ** 2)
        centered = X - X.mean(axis=1, keepdims=True)
        slope = np.sum(centered * t_centered[None, :, None], axis=1) / denom

        return np.concatenate(
            [mean, std, min_v, max_v, p25, p50, p75, slope],
            axis=1,
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """
        Train the Random Forest model.
        
        Args:
            X: Training features (n_samples, window_size, n_features)
            y: Training labels (n_samples,)
        
        Returns:
            self
        """
        X_flat = self._extract_window_features(X)
        self.n_features_in_ = X_flat.shape[1]
        
        print(f"Training Random Forest on {X_flat.shape[0]} samples, {X_flat.shape[1]} features...")
        self.model.fit(X_flat, y)
        self.is_fitted = True
        print("Training complete.")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fault labels.
        
        Args:
            X: Features (n_samples, window_size, n_features)
        
        Returns:
            Predicted labels (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_flat = self._extract_window_features(X)
        return self.model.predict(X_flat)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fault probabilities.
        
        Args:
            X: Features (n_samples, window_size, n_features)
        
        Returns:
            Predicted probabilities (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_flat = self._extract_window_features(X)
        return self.model.predict_proba(X_flat)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y: True labels
        
        Returns:
            Dict of evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]  # Probability of fault class
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1_score": f1_score(y, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: Optional list of feature names
        
        Returns:
            Dict mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return dict(zip(feature_names, importances))
    
    def save(self, path: Optional[Path] = None, name: str = "random_forest") -> Path:
        """
        Save model to disk.
        
        Args:
            path: Directory to save to (defaults to MODELS_DIR)
            name: Model filename (without extension)
        
        Returns:
            Path to saved model
        """
        if path is None:
            path = MODELS_DIR
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        filepath = path / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to: {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> "RandomForestModel":
        """
        Load model from disk.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return model


if __name__ == "__main__":
    # Quick test
    from src.preprocessing import load_processed_data
    
    print("Testing Random Forest model...")
    data = load_processed_data("FD002")
    
    X_train = data['X_train'][:1000]  # Subset for quick test
    y_train = data['y_train'][:1000]
    X_test = data['X_test'][:500]
    y_test = data['y_test'][:500]
    
    model = RandomForestModel()
    model.fit(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    print(f"\nTest Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
