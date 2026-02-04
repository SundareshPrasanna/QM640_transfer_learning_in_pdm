"""
Run Fine-Tuning Experiments (RQ2)

Fine-tunes CNN and LSTM on target domains and compares with direct transfer.
Tests hypothesis H02 vs H12.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

from src.config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, REPORTS_DIR,
    SOURCE_DOMAIN, TARGET_DOMAINS, ALPHA, FINE_TUNING_PARAMS
)
from src.transfer_learning import run_fine_tuning_experiment, load_pretrained_model
from src.preprocessing import load_processed_data
from src.evaluation import evaluate_torch_model
from src.data_loader import CMAPSSDataset
from torch.utils.data import DataLoader


def load_direct_transfer_results():
    """Load results from RQ1 domain shift analysis."""
    results_path = REPORTS_DIR / "rq1_results.csv"
    if results_path.exists():
        return pd.read_csv(results_path)
    else:
        print("Warning: RQ1 results not found. Run domain shift analysis first.")
        return None


def run_fine_tuning_experiments():
    """Run all fine-tuning experiments for RQ2."""
    print("="*60)
    print("FINE-TUNING EXPERIMENTS (RQ2)")
    print(f"Source: {SOURCE_DOMAIN} → Targets: {TARGET_DOMAINS}")
    print(f"Label Fraction: 20%")
    print(f"Device: {DEVICE}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Load direct transfer results for comparison
    direct_results = load_direct_transfer_results()
    
    results = []
    
    for model_type in ['cnn', 'lstm']:
        for target_domain in TARGET_DOMAINS:
            # Run fine-tuning experiment
            exp_result = run_fine_tuning_experiment(
                model_type=model_type,
                target_domain=target_domain,
                label_fraction=0.2,  # 20% labeled target data
                freeze_base=True,
            )
            
            # Get direct transfer result for comparison
            if direct_results is not None:
                direct_row = direct_results[
                    (direct_results['model'] == model_type) & 
                    (direct_results['domain'] == target_domain)
                ]
                if not direct_row.empty:
                    direct_f1 = direct_row['f1_score'].values[0]
                else:
                    direct_f1 = None
            else:
                direct_f1 = None
            
            fine_tuned_f1 = exp_result['test_metrics']['f1_score']
            improvement = fine_tuned_f1 - direct_f1 if direct_f1 is not None else None
            
            results.append({
                'model': model_type,
                'target_domain': target_domain,
                'direct_transfer_f1': direct_f1,
                'fine_tuned_f1': fine_tuned_f1,
                'improvement': improvement,
                'accuracy': exp_result['test_metrics']['accuracy'],
                'roc_auc': exp_result['test_metrics']['roc_auc'],
            })
    
    return pd.DataFrame(results)


def statistical_hypothesis_test(results_df):
    """
    Test H02 vs H12: Does fine-tuning significantly improve performance?
    
    H02: Fine-tuning does not significantly improve performance
    H12: Fine-tuning significantly improves performance
    """
    print("\n" + "="*60)
    print("STATISTICAL HYPOTHESIS TESTING (H02 vs H12)")
    print("="*60)
    print(f"\nSignificance level α = {ALPHA}")
    print("\nH02: Fine-tuning does not significantly improve performance")
    print("H12: Fine-tuning significantly improves performance")
    
    test_results = []
    
    for model_type in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model_type]
        
        direct_f1s = model_df['direct_transfer_f1'].values
        fine_tuned_f1s = model_df['fine_tuned_f1'].values
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(fine_tuned_f1s, direct_f1s)
        
        # One-sided test: fine-tuned > direct
        one_sided_p = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        
        significant = one_sided_p < ALPHA and t_stat > 0
        
        mean_improvement = np.mean(fine_tuned_f1s - direct_f1s)
        
        print(f"\n{'-'*40}")
        print(f"Model: {model_type.upper()}")
        print(f"  Mean Direct Transfer F1: {np.mean(direct_f1s):.4f}")
        print(f"  Mean Fine-Tuned F1: {np.mean(fine_tuned_f1s):.4f}")
        print(f"  Mean Improvement: {mean_improvement:+.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value (one-sided): {one_sided_p:.4f}")
        print(f"  Result: {'REJECT H02 (significant improvement)' if significant else 'FAIL TO REJECT H02'}")
        
        test_results.append({
            'model': model_type,
            'mean_direct_f1': np.mean(direct_f1s),
            'mean_finetuned_f1': np.mean(fine_tuned_f1s),
            'mean_improvement': mean_improvement,
            't_statistic': t_stat,
            'p_value': one_sided_p,
            'significant': significant,
        })
    
    return pd.DataFrame(test_results)


def generate_report(results_df, stats_df):
    """Generate and save RQ2 analysis report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    report_path = REPORTS_DIR / "rq2_fine_tuning_analysis.md"
    
    with open(report_path, 'w') as f:
        f.write("# RQ2: Fine-Tuning Analysis Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Research Question\n\n")
        f.write("**RQ2:** To what extent does fine-tuning improve fault detection ")
        f.write("performance compared to direct transfer?\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write(f"- **Source Domain:** {SOURCE_DOMAIN}\n")
        f.write(f"- **Target Domains:** {', '.join(TARGET_DOMAINS)}\n")
        f.write("- **Fine-tuning Strategy:** Freeze base layers, train classification head\n")
        f.write("- **Labeled Target Data:** 20%\n")
        f.write(f"- **Significance Level:** α = {ALPHA}\n\n")
        
        f.write("## Results\n\n")
        f.write("### Performance Comparison\n\n")
        f.write("| Model | Target | Direct F1 | Fine-Tuned F1 | Improvement |\n")
        f.write("|-------|--------|-----------|---------------|-------------|\n")
        
        for _, row in results_df.iterrows():
            imp = f"+{row['improvement']:.4f}" if row['improvement'] >= 0 else f"{row['improvement']:.4f}"
            f.write(f"| {row['model']} | {row['target_domain']} | ")
            f.write(f"{row['direct_transfer_f1']:.4f} | {row['fine_tuned_f1']:.4f} | {imp} |\n")
        
        f.write("\n### Statistical Test Results\n\n")
        f.write("| Model | Mean Direct F1 | Mean Fine-Tuned F1 | Improvement | p-value | Significant |\n")
        f.write("|-------|----------------|---------------------|-------------|---------|-------------|\n")
        
        for _, row in stats_df.iterrows():
            sig = "✓" if row['significant'] else "✗"
            f.write(f"| {row['model']} | {row['mean_direct_f1']:.4f} | ")
            f.write(f"{row['mean_finetuned_f1']:.4f} | {row['mean_improvement']:+.4f} | ")
            f.write(f"{row['p_value']:.4f} | {sig} |\n")
        
        f.write("\n## Hypothesis Test Results\n\n")
        f.write("**H02:** Fine-tuning does not significantly improve performance.\n\n")
        f.write("**H12:** Fine-tuning significantly improves performance.\n\n")
        
        # Overall conclusion
        significant_models = stats_df[stats_df['significant']]['model'].tolist()
        if significant_models:
            f.write(f"**Conclusion:** We REJECT H02 for {', '.join(significant_models)}. ")
            f.write("Fine-tuning provides statistically significant improvement.\n")
        else:
            f.write("**Conclusion:** We FAIL TO REJECT H02. Fine-tuning does not ")
            f.write("provide statistically significant improvement.\n")
    
    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    """Main function to run RQ2 experiments."""
    # Run fine-tuning experiments
    results_df = run_fine_tuning_experiments()
    
    # Print summary
    print("\n" + "="*60)
    print("FINE-TUNING RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Statistical testing
    stats_df = statistical_hypothesis_test(results_df)
    
    # Generate report
    report_path = generate_report(results_df, stats_df)
    
    # Save raw results
    results_path = REPORTS_DIR / "rq2_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("FINE-TUNING EXPERIMENTS COMPLETE (RQ2)")
    print("="*60)
    
    return results_df, stats_df


if __name__ == "__main__":
    results_df, stats_df = main()
