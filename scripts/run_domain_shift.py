"""
Run Domain Shift Analysis (RQ1)

Evaluates source-trained models on target domains and quantifies performance degradation.
Tests hypothesis H01 vs H11.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from scipy import stats

from src.config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, REPORTS_DIR,
    SOURCE_DOMAIN, TARGET_DOMAINS, ALPHA
)
from src.preprocessing import load_processed_data
from src.data_loader import CMAPSSDataset
from src.models.random_forest import RandomForestModel
from src.models.cnn import CNNModel
from src.models.lstm import LSTMModel
from src.evaluation import (
    evaluate_sklearn_model, evaluate_torch_model,
    compute_metrics, generate_domain_shift_report
)
from torch.utils.data import DataLoader


def load_all_models():
    """Load all trained baseline models."""
    models = {}
    
    # Random Forest
    rf_path = MODELS_DIR / f"rf_{SOURCE_DOMAIN}.pkl"
    if rf_path.exists():
        models['random_forest'] = {
            'model': RandomForestModel.load(rf_path),
            'is_torch': False,
        }
        print(f"Loaded: {rf_path}")
    
    # CNN
    cnn_path = MODELS_DIR / f"cnn_{SOURCE_DOMAIN}.pt"
    if cnn_path.exists():
        # Get number of features from source domain
        source_data = load_processed_data(SOURCE_DOMAIN)
        n_features = len(source_data['feature_cols'])
        
        cnn = CNNModel(input_channels=n_features)
        checkpoint = torch.load(cnn_path, map_location=DEVICE)
        cnn.model.load_state_dict(checkpoint['model_state_dict'])
        cnn.model.eval()
        
        models['cnn'] = {
            'model': cnn.model,
            'is_torch': True,
            'n_features': n_features,
        }
        print(f"Loaded: {cnn_path}")
    
    # LSTM
    lstm_path = MODELS_DIR / f"lstm_{SOURCE_DOMAIN}.pt"
    if lstm_path.exists():
        source_data = load_processed_data(SOURCE_DOMAIN)
        n_features = len(source_data['feature_cols'])
        
        lstm = LSTMModel(input_size=n_features)
        checkpoint = torch.load(lstm_path, map_location=DEVICE)
        lstm.model.load_state_dict(checkpoint['model_state_dict'])
        lstm.model.eval()
        
        models['lstm'] = {
            'model': lstm.model,
            'is_torch': True,
            'n_features': n_features,
        }
        print(f"Loaded: {lstm_path}")
    
    return models


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


def evaluate_all_domains(models, source_domain=SOURCE_DOMAIN, target_domains=TARGET_DOMAINS):
    """Evaluate all models on all domains."""
    results = []
    source_data = load_processed_data(source_domain)
    source_features = source_data['feature_cols']
    
    all_domains = [source_domain] + target_domains
    
    for model_name, model_info in models.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name.upper()}")
        print(f"{'='*50}")
        
        for domain in all_domains:
            data = load_processed_data(domain)
            target_features = data['feature_cols']
            
            # Align features if needed
            if len(target_features) != len(source_features):
                X_test = align_features_for_transfer(
                    data['X_test'], target_features, source_features
                )
                print(f"  {domain}: Aligned features ({len(target_features)} -> {len(source_features)})")
            else:
                X_test = data['X_test']
            
            y_test = data['y_test']
            
            if model_info['is_torch']:
                dataset = CMAPSSDataset(X_test, y_test)
                loader = DataLoader(dataset, batch_size=64, shuffle=False)
                metrics = evaluate_torch_model(model_info['model'], loader)
            else:
                metrics = evaluate_sklearn_model(model_info['model'], X_test, y_test)
            
            metrics['model'] = model_name
            metrics['domain'] = domain
            metrics['domain_type'] = 'source' if domain == source_domain else 'target'
            results.append(metrics)
            
            print(f"  {domain}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}, AUC={metrics['roc_auc']:.4f}")
    
    return pd.DataFrame(results)


def statistical_hypothesis_test(results_df):
    """
    Test H01 vs H11: Is there significant performance degradation under domain shift?
    
    H01: No significant difference between source and target performance
    H11: Significant decrease in performance under direct transfer
    """
    print("\n" + "="*60)
    print("STATISTICAL HYPOTHESIS TESTING (H01 vs H11)")
    print("="*60)
    print(f"\nSignificance level α = {ALPHA}")
    print("\nH01: No significant difference in fault detection accuracy")
    print("H11: Significant decrease in accuracy under direct transfer")
    
    test_results = []
    
    for model_name in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model_name]
        
        source_row = model_df[model_df['domain_type'] == 'source'].iloc[0]
        target_rows = model_df[model_df['domain_type'] == 'target']
        
        source_f1 = source_row['f1_score']
        target_f1s = target_rows['f1_score'].values
        
        # Compute mean degradation
        mean_target_f1 = np.mean(target_f1s)
        degradation = source_f1 - mean_target_f1
        
        # One-sample t-test: Is target F1 significantly lower than source F1?
        t_stat, p_value = stats.ttest_1samp(target_f1s, source_f1)
        
        # For one-sided test (we expect degradation), divide p-value by 2
        # and check if t_stat is negative (target < source)
        one_sided_p = p_value / 2 if t_stat < 0 else 1 - p_value / 2
        
        significant = one_sided_p < ALPHA and t_stat < 0
        
        print(f"\n{'-'*40}")
        print(f"Model: {model_name.upper()}")
        print(f"  Source F1: {source_f1:.4f}")
        print(f"  Mean Target F1: {mean_target_f1:.4f}")
        print(f"  Degradation: {degradation:+.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value (one-sided): {one_sided_p:.4f}")
        print(f"  Result: {'REJECT H01 (significant degradation)' if significant else 'FAIL TO REJECT H01'}")
        
        test_results.append({
            'model': model_name,
            'source_f1': source_f1,
            'mean_target_f1': mean_target_f1,
            'degradation': degradation,
            't_statistic': t_stat,
            'p_value': one_sided_p,
            'significant': significant,
            'conclusion': 'Significant degradation' if significant else 'No significant degradation',
        })
    
    return pd.DataFrame(test_results)


def generate_report(results_df, stats_df):
    """Generate and save analysis report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report_path = REPORTS_DIR / "rq1_domain_shift_analysis.md"
    
    with open(report_path, 'w') as f:
        f.write("# RQ1: Domain Shift Analysis Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Research Question\n\n")
        f.write("**RQ1:** How does the performance of a fault detection model degrade when transferred ")
        f.write("directly from a large source dataset to smaller target datasets without adaptation?\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write(f"- **Source Domain:** {SOURCE_DOMAIN}\n")
        f.write(f"- **Target Domains:** {', '.join(TARGET_DOMAINS)}\n")
        f.write("- **Transfer Method:** Direct transfer (no adaptation)\n")
        f.write(f"- **Significance Level:** α = {ALPHA}\n\n")
        
        f.write("## Results\n\n")
        f.write("### Performance by Domain\n\n")
        f.write("| Model | Domain | Type | Accuracy | F1-Score | ROC-AUC |\n")
        f.write("|-------|--------|------|----------|----------|----------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['model']} | {row['domain']} | {row['domain_type']} | ")
            f.write(f"{row['accuracy']:.4f} | {row['f1_score']:.4f} | {row['roc_auc']:.4f} |\n")
        
        f.write("\n### Performance Degradation Summary\n\n")
        f.write("| Model | Source F1 | Mean Target F1 | Degradation | p-value | Significant |\n")
        f.write("|-------|-----------|----------------|-------------|---------|-------------|\n")
        
        for _, row in stats_df.iterrows():
            sig = "✓" if row['significant'] else "✗"
            f.write(f"| {row['model']} | {row['source_f1']:.4f} | {row['mean_target_f1']:.4f} | ")
            f.write(f"{row['degradation']:+.4f} | {row['p_value']:.4f} | {sig} |\n")
        
        f.write("\n## Hypothesis Test Results\n\n")
        f.write("**H01:** There is no statistically significant difference in fault detection ")
        f.write("accuracy between source-trained model and its performance on target datasets.\n\n")
        f.write("**H11:** There is a statistically significant decrease in fault detection ")
        f.write("accuracy under direct transfer.\n\n")
        
        # Overall conclusion
        significant_models = stats_df[stats_df['significant']]['model'].tolist()
        if significant_models:
            f.write(f"**Conclusion:** We REJECT H01 for {', '.join(significant_models)}. ")
            f.write("There is statistically significant performance degradation under domain shift.\n")
        else:
            f.write("**Conclusion:** We FAIL TO REJECT H01. No statistically significant ")
            f.write("performance degradation was observed.\n")
    
    print(f"\nReport saved to: {report_path}")
    return report_path


def run_domain_shift_analysis():
    """Main function to run complete domain shift analysis."""
    print("="*60)
    print("DOMAIN SHIFT ANALYSIS (RQ1)")
    print(f"Source: {SOURCE_DOMAIN} → Targets: {TARGET_DOMAINS}")
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Load models
    print("\nLoading trained models...")
    models = load_all_models()
    
    if not models:
        print("ERROR: No trained models found!")
        return None, None
    
    # Evaluate on all domains
    results_df = evaluate_all_domains(models)
    
    # Statistical hypothesis testing
    stats_df = statistical_hypothesis_test(results_df)
    
    # Generate report
    report_path = generate_report(results_df, stats_df)
    
    # Save raw results
    results_path = REPORTS_DIR / "rq1_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("DOMAIN SHIFT ANALYSIS COMPLETE")
    print("="*60)
    
    return results_df, stats_df


if __name__ == "__main__":
    results_df, stats_df = run_domain_shift_analysis()
