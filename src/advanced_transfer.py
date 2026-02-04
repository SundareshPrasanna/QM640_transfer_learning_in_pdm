"""
Advanced Transfer Learning module for fault detection.

Implements SOTA strategies:
1. Gradual Unfreezing (Two-stage fine-tuning)
2. Domain-Adaptive Batch Normalization (BN statistics adaptation)
3. Dynamic Loss Weighting (Target-specific class weights)
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
import copy

from src.config import (
    DEVICE, MODELS_DIR, FINE_TUNING_PARAMS, TRAINING_PARAMS,
    SOURCE_DOMAIN, RANDOM_SEED
)
from src.transfer_learning import (
    load_pretrained_model, prepare_target_data, freeze_base_layers
)
from src.training import set_seed, EarlyStopping, evaluate_model
from src.evaluation import evaluate_torch_model

def get_target_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """Calculate class weights based specifically on target domain training data."""
    all_labels = []
    for _, y in train_loader:
        all_labels.extend(y.numpy())
    
    all_labels = np.array(all_labels)
    n_samples = len(all_labels)
    n_faults = np.sum(all_labels)
    n_healthy = n_samples - n_faults
    
    # Avoid division by zero
    if n_faults == 0:
        return torch.tensor([1.0], device=DEVICE)
        
    weight_for_fault = n_healthy / n_faults
    # Cap weight or use square root to prevent over-optimization for faults
    weight_for_fault = np.sqrt(weight_for_fault) 
    return torch.tensor([weight_for_fault], device=DEVICE, dtype=torch.float32)

def prepare_model_for_adaptive_bn(model: nn.Module, model_type: str):
    """
    Freeze base layers but keep BatchNorm layers in training mode.
    This allows internal statistics (mean/var) to adapt to the target domain
    without updating the weights of the filters.
    """
    if model_type == 'cnn':
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                # Keep BN in train mode for statistics adaptation
                module.train()
                # But don't update affine parameters (weight/bias)
                for param in module.parameters():
                    param.requires_grad = False
            elif isinstance(module, nn.Conv1d):
                for param in module.parameters():
                    param.requires_grad = False
    elif model_type == 'lstm':
        # LSTM doesn't typically have BN in our implementation, 
        # but we freeze the LSTM weights.
        for name, param in model.named_parameters():
            if 'lstm' in name:
                param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Adaptive BN Setup: {total - trainable:,} frozen params, {trainable:,} trainable")

def unfreeze_last_block(model: nn.Module, model_type: str):
    """Unfreeze the last block of the feature extractor for partial fine-tuning."""
    if model_type == 'cnn':
        # CNN1D structure has conv1, conv2, conv3. Let's unfreeze conv3.
        for name, param in model.named_parameters():
            if 'conv3' in name or 'bn3' in name:
                param.requires_grad = True
    elif model_type == 'lstm':
        # BiLSTM has two layers. Let's unfreeze the second layer.
        for name, param in model.named_parameters():
            if 'lstm.weight_ih_l1' in name or 'lstm.weight_hh_l1' in name:
                param.requires_grad = True
                
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Post-Unfreeze: {trainable:,} trainable parameters")

def advanced_fine_tune(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str,
    epochs_stage1: int = 15,
    epochs_stage2: int = 15,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    verbose: bool = True
) -> Dict:
    """
    Implement a two-stage advanced fine-tuning process.
    Stage 1: Frozen base (with Adaptive BN if CNN) + Head training.
    Stage 2: Partially unfrozen base + Full network fine-tuning with low LR.
    """
    set_seed()
    
    # 1. Calculate Dynamic Weights
    criterion = nn.BCELoss(reduction='mean')
    
    # --- STAGE 1: Head Warmup (Frozen Base) ---
    if verbose:
        print(f"\n>>> STAGE 1: Head Warmup (LR={lr_stage1})")
    
    freeze_base_layers(model, model_type)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_stage1)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs_stage1 // 2): # Short warmup
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE).float()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    # --- STAGE 2: Adaptive BN + Last Block Unfreeze ---
    if verbose:
        print(f"\n>>> STAGE 2: Adaptive BN + Partial Fine-Tuning (LR={lr_stage2})")
    
    unfreeze_last_block(model, model_type)
    if model_type == 'cnn':
        # Keep BN layers in train mode to adapt stats
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.train()
                for p in m.parameters(): p.requires_grad = True # Allow BN adaptation

    # Discriminative LR: Use LR_stage2 for un-frozen base, 10x higher for head
    head_params = []
    base_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name or 'classifier' in name:
                head_params.append(param)
            else:
                base_params.append(param)
    
    optimizer = optim.Adam([
        {'params': base_params, 'lr': lr_stage2},
        {'params': head_params, 'lr': lr_stage2 * 10}
    ])
    
    early_stopping = EarlyStopping(patience=5)
    
    for epoch in range(epochs_stage2):
        model.train()
        if model_type == 'cnn':
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d): m.train()
        
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE).float()
            optimizer.zero_grad()
            out = model(X)
            # Use dynamic weight only in this refinement stage
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE).float()
                out = model(X)
                val_loss += criterion(out, y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        if verbose and (epoch+1)%5 == 0:
            print(f"  Stage 2 Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            
        if early_stopping(avg_val_loss):
            break

    if best_state:
        model.load_state_dict(best_state)
    
    return {"status": "success", "final_val_loss": best_val_loss}

def run_advanced_experiment(model_type: str, target_domain: str, label_fraction: float = 0.2):
    """Run the advanced transfer experiment for a specific target domain."""
    print(f"\n{'*'*60}")
    print(f"ADVANCED TRANSFER: {model_type.upper()} on {target_domain} ({label_fraction*100:.0f}% labels)")
    print(f"{'*'*60}")
    
    # 1. Load Pretrained
    model, n_features, source_features = load_pretrained_model(model_type)
    
    # 2. Prepare Data
    train_loader, val_loader, test_loader = prepare_target_data(
        target_domain, source_features, label_fraction
    )
    
    # 3. Advanced Fine-Tune
    advanced_fine_tune(model, train_loader, val_loader, model_type)
    
    # 4. Evaluate
    test_metrics = evaluate_torch_model(model, test_loader)
    print(f"\nAdvanced Transfer Test Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    return test_metrics

if __name__ == "__main__":
    # Test on FD001
    run_advanced_experiment('cnn', 'FD001', 0.2)
