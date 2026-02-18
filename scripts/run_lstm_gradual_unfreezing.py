"""
LSTM Gradual Unfreezing Transfer Comparison Script.

Runs LSTM-only gradual unfreezing transfer and compares against LSTM base
fine-tuning (RQ2).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime
from src.config import REPORTS_DIR, TARGET_DOMAINS
from src.advanced_transfer import run_advanced_experiment

def compare_lstm_gradual_unfreezing():
    """Run LSTM gradual unfreezing and compare with LSTM baseline fine-tuning."""
    print("="*60)
    print("LSTM GRADUAL UNFREEZING TRANSFER ANALYSIS")
    print("="*60)
    
    # 1. Load base LSTM fine-tuning results from RQ2.
    base_results_path = REPORTS_DIR / "rq2_results.csv"
    if not base_results_path.exists():
        print(f"Error: Base results not found at {base_results_path}")
        return
    
    base_df = pd.read_csv(base_results_path)
    base_df = base_df[base_df['model'] == 'lstm']
    if base_df.empty:
        print("Error: No LSTM rows found in rq2_results.csv")
        return
    
    advanced_results = []
    
    # 2. Run LSTM-only gradual unfreezing experiment per target domain.
    for target in TARGET_DOMAINS:
        adv_metrics = run_advanced_experiment('lstm', target, label_fraction=0.2)
        
        # Find base LSTM fine-tuned F1 for comparison
        base_row = base_df[base_df['target_domain'] == target]
        base_f1 = base_row['fine_tuned_f1'].values[0] if not base_row.empty else 0
        
        advanced_results.append({
            'model': 'lstm',
            'target_domain': target,
            'base_f1': base_f1,
            'advanced_f1': adv_metrics['f1_score'],
            'improvement': adv_metrics['f1_score'] - base_f1,
            'pct_improvement': ((adv_metrics['f1_score'] - base_f1) / base_f1 * 100) if base_f1 > 0 else 0
        })
    
    results_df = pd.DataFrame(advanced_results)
    
    # 3. Generate comparative report.
    report_path = REPORTS_DIR / "lstm_gradual_unfreezing_report.md"
    with open(report_path, 'w') as f:
        f.write("# LSTM Gradual Unfreezing Transfer Analysis\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write("- **Model scope:** LSTM only (no CNN in this analysis).\n")
        f.write("- **Stage 1 (head warmup):** Freeze LSTM backbone and train classification head.\n")
        f.write("- **Stage 2 (gradual unfreezing):** Unfreeze the upper LSTM layer and fine-tune with lower LR.\n")
        f.write("- **Class balancing:** Use target-domain class-weighted BCE during adaptation.\n\n")
        
        f.write("## Performance Comparison (F1-Score)\n\n")
        f.write("| Target | Base LSTM FT (RQ2) | LSTM Gradual Unfreezing | Gain | % Gain |\n")
        f.write("|--------|--------------------|--------------------------|------|--------|\n")
        
        for _, row in results_df.iterrows():
            f.write(
                f"| {row['target_domain']} | {row['base_f1']:.4f} | "
                f"{row['advanced_f1']:.4f} | {row['improvement']:+.4f} | "
                f"{row['pct_improvement']:+.1f}% |\n"
            )
            
        f.write("\n## Key Takeaways\n\n")
        avg_gain = results_df['improvement'].mean()
        f.write(f"- Average F1-score improvement across target domains: **{avg_gain:+.4f}**\n")
        
        max_gain_row = results_df.loc[results_df['improvement'].idxmax()]
        f.write(
            f"- Maximum gain observed: **{max_gain_row['improvement']:+.4f}** "
            f"(on {max_gain_row['target_domain']})\n\n"
        )
        
        f.write(
            "This result isolates the impact of **LSTM gradual unfreezing** against "
            "standard LSTM fine-tuning on the same target splits.\n"
        )

    print(f"\nComparative report saved to: {report_path}")
    csv_path = REPORTS_DIR / "lstm_gradual_unfreezing_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Results table saved to: {csv_path}")

if __name__ == "__main__":
    compare_lstm_gradual_unfreezing()
