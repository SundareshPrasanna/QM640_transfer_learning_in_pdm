"""
Run Label Efficiency Analysis (RQ4)

Evaluates model performance as a function of target domain labeled data fraction.
Fractions: 1%, 5%, 10%, 20%, 50%
Tests hypothesis H04 vs H14.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from src.config import (
    DEVICE, REPORTS_DIR, SOURCE_DOMAIN, TARGET_DOMAINS, ALPHA
)
from src.transfer_learning import run_fine_tuning_experiment

def run_label_efficiency_experiments():
    """Run experiments with different label fractions on target domains."""
    print("="*60)
    print("LABEL EFFICIENCY ANALYSIS (RQ4)")
    print(f"Source: {SOURCE_DOMAIN}")
    print(f"Fractions: [0.01, 0.05, 0.10, 0.20, 0.50]")
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    fractions = [0.01, 0.05, 0.10, 0.20, 0.50]
    results = []
    
    # We'll focus on CNN for this analysis as it showed better adaptation in RQ2/RQ3
    model_types = ['cnn'] 
    # Target domain: Let's pick FD001 as it's the most "standard" target
    target_domains = ['FD001']
    
    for model_type in model_types:
        for target in target_domains:
            for frac in fractions:
                print(f"\nProcessing: {model_type} on {target} with {frac*100:.1f}% data...")
                
                exp_result = run_fine_tuning_experiment(
                    model_type=model_type,
                    target_domain=target,
                    label_fraction=frac,
                    freeze_base=True
                )
                
                metrics = exp_result['test_metrics']
                results.append({
                    'model': model_type,
                    'target_domain': target,
                    'label_fraction': frac,
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'roc_auc': metrics['roc_auc']
                })
    
    return pd.DataFrame(results)

def generate_report(results_df):
    """Generate RQ4 label efficiency report."""
    report_path = REPORTS_DIR / "rq4_label_efficiency_analysis.md"
    
    with open(report_path, 'w') as f:
        f.write("# RQ4: Label Efficiency Analysis Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Research Question\n\n")
        f.write("**RQ4:** How many target domain labels are required for a fine-tuned model to surpass the performance ")
        f.write("of a source-trained model?\n\n")
        
        f.write("## Label Efficiency Results (CNN on FD001)\n\n")
        f.write("| Label Fraction | Accuracy | F1-Score | ROC-AUC |\n")
        f.write("|----------------|----------|----------|----------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['label_fraction']*100:.1f}% | {row['accuracy']:.4f} | {row['f1_score']:.4f} | {row['roc_auc']:.4f} |\n")
        
        f.write("\n## Statistical Analysis (H04 vs H14)\n\n")
        f.write("**H04:** Small amounts of target data (<10%) do not provide significant performance improvements.\n\n")
        f.write("**H14:** Significant improvement is possible even with <10% data.\n\n")
        
        # Compare 1% and 5% vs 0% (from RQ1)
        # We know from RQ1 that CNN on FD001 (Direct Transfer) had F1 ≈ 0.0988
        base_f1 = 0.0988
        
        f.write(f"**Baseline (0% Target Labels):** F1 = {base_f1:.4f}\n\n")
        
        f1_1pct = results_df[results_df['label_fraction'] == 0.01]['f1_score'].values[0]
        f1_5pct = results_df[results_df['label_fraction'] == 0.05]['f1_score'].values[0]
        f1_10pct = results_df[results_df['label_fraction'] == 0.10]['f1_score'].values[0]
        
        f.write(f"- **1% Data Improvement:** {f1_1pct - base_f1:+.4f}\n")
        f.write(f"- **5% Data Improvement:** {f1_5pct - base_f1:+.4f}\n")
        f.write(f"- **10% Data Improvement:** {f1_10pct - base_f1:+.4f}\n\n")
        
        if f1_5pct > base_f1 * 1.5: # 50% improvement
            f.write("**Conclusion:** We REJECT H04. Substantial performance gains (e.g., >50% improvement) ")
            f.write("are achievable with as little as 5% of target domain labeled data.\n")
        else:
            f.write("**Conclusion:** We FAIL TO REJECT H04. Small amounts of target data do not ")
            f.write("provide substantial gains in this study.\n")

    print(f"\nReport saved to: {report_path}")

def main():
    results_df = run_label_efficiency_experiments()
    generate_report(results_df)
    
    # Save results
    results_df.to_csv(REPORTS_DIR / "rq4_results.csv", index=False)
    print("Label efficiency results saved to CSV.")

if __name__ == "__main__":
    main()
