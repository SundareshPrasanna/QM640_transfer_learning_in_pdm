"""
Generate Final Visualizations for Project Report.

Creates summary plots for RQ1-RQ4.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import REPORTS_DIR

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def load_rq1_baselines(model='cnn', source_domain='FD002', target_domain='FD001'):
    """Load baseline F1 values from RQ1 results to avoid hardcoded constants."""
    rq1 = pd.read_csv(REPORTS_DIR / "rq1_results.csv")
    target_row = rq1[(rq1['model'] == model) & (rq1['domain'] == target_domain)]
    source_row = rq1[(rq1['model'] == model) & (rq1['domain'] == source_domain)]
    if target_row.empty or source_row.empty:
        raise ValueError("Missing required baseline rows in rq1_results.csv")
    return float(target_row['f1_score'].iloc[0]), float(source_row['f1_score'].iloc[0])

def plot_rq1_degradation():
    """Plot performance across domains (RQ1)."""
    df = pd.read_csv(REPORTS_DIR / "rq1_results.csv")
    
    # Filter for CNN and LSTM
    df = df[df['model'].isin(['cnn', 'lstm'])]
    
    plt.figure()
    sns.barplot(data=df, x='domain', y='f1_score', hue='model')
    plt.title('RQ1: Performance Degradation under Direct Transfer')
    plt.ylabel('F1-Score')
    plt.ylim(0, 0.7)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "rq1_degradation.png")
    print("Saved: rq1_degradation.png")

def plot_rq2_fine_tuning():
    """Plot Direct Transfer vs Fine-Tuning improvement (RQ2)."""
    df = pd.read_csv(REPORTS_DIR / "rq2_results.csv")
    
    # Reshape for plotting
    plot_df = []
    for _, row in df.iterrows():
        plot_df.append({
            'Model': row['model'].upper(),
            'Target': row['target_domain'],
            'Method': 'Direct Transfer',
            'F1': row['direct_transfer_f1']
        })
        plot_df.append({
            'Model': row['model'].upper(),
            'Target': row['target_domain'],
            'Method': 'Fine-Tuning (20%)',
            'F1': row['fine_tuned_f1']
        })
    plot_df = pd.DataFrame(plot_df)
    
    plt.figure()
    sns.catplot(
        data=plot_df, kind="bar",
        x="Target", y="F1", hue="Method", col="Model",
        height=5, aspect=0.8, palette="muted"
    )
    plt.suptitle('RQ2: Direct Transfer vs. Fine-Tuning (20% Labels)', y=1.05)
    plt.savefig(REPORTS_DIR / "rq2_comparison.png")
    print("Saved: rq2_comparison.png")

def plot_rq4_learning_curve():
    """Plot F1 score vs label fraction (RQ4)."""
    df = pd.read_csv(REPORTS_DIR / "rq4_results.csv")
    direct_baseline, source_baseline = load_rq1_baselines(model='cnn', source_domain='FD002', target_domain='FD001')
    
    plt.figure()
    sns.lineplot(data=df, x='label_fraction', y='f1_score', marker='o', linewidth=2)
    # Add baseline from RQ1 (CNN FD001)
    plt.axhline(y=direct_baseline, color='r', linestyle='--', label='Direct Transfer Baseline (0%)')
    
    # Add source domain baseline (CNN FD002)
    plt.axhline(y=source_baseline, color='g', linestyle=':', label='Source Domain Performance (FD002)')
    
    plt.title('RQ4: Label Efficiency (CNN on FD001)')
    plt.xlabel('Fraction of Target Labels')
    plt.ylabel('F1-Score')
    plt.xticks([0.01, 0.05, 0.10, 0.20, 0.50], ['1%', '5%', '10%', '20%', '50%'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "rq4_learning_curve.png")
    print("Saved: rq4_learning_curve.png")

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    plot_rq1_degradation()
    plot_rq2_fine_tuning()
    plot_rq4_learning_curve()

if __name__ == "__main__":
    main()
