"""
Generate detailed data description report with statistics and visualizations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_raw_data, identify_constant_sensors
from src.config import DATASETS, SENSORS_TO_USE

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

OUTPUT_DIR = Path("results")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def calculate_statistics(df, sensors):
    """Calculate descriptive statistics for sensor columns."""
    stats = df[sensors].describe().T
    
    # Add missing values count and percentage
    missing_count = df[sensors].isnull().sum()
    missing_pct = 100 * df[sensors].isnull().sum() / len(df)
    
    stats['Missing Count'] = missing_count
    stats['Missing %'] = missing_pct
    
    # Add Variance
    stats['Variance'] = df[sensors].var()
    
    # Reorder columns
    cols = ['count', 'mean', 'std', 'Variance', 'min', '25%', '50%', '75%', 'max', 'Missing Count', 'Missing %']
    return stats[cols]

def plot_distributions(df, dataset_name, sensors):
    """Generate histograms for sensor distributions."""
    # Plot top 4 sensors with highest variance to keep report concise, or all if few
    # For now, let's plot a subset of important sensors to avoid overcrowding
    selected_sensors = sensors[:4] # Just taking first 4 for brevity in this example, or we can do all. 
                                    # Let's do a grid of all active sensors.
    
    n_sensors = len(sensors)
    n_cols = 3
    n_rows = (n_sensors + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 4 * n_rows))
    for i, sensor in enumerate(sensors):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(data=df, x=sensor, kde=True, bins=30)
        plt.title(f'{sensor} Distribution')
        plt.tight_layout()
        
    filename = FIGURES_DIR / f"{dataset_name}_distributions.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename.name

def plot_boxplots(df, dataset_name, sensors):
    """Generate boxplots to visualize outliers."""
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df[sensors], orient='h')
    plt.title(f'{dataset_name} - Sensor Boxplots')
    plt.tight_layout()
    
    filename = FIGURES_DIR / f"{dataset_name}_boxplots.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename.name

def plot_correlation_heatmap(df, dataset_name, sensors):
    """Generate correlation heatmap."""
    plt.figure(figsize=(12, 10))
    corr = df[sensors].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True)
    plt.title(f'{dataset_name} - Sensor Correlation Matrix')
    plt.tight_layout()
    
    filename = FIGURES_DIR / f"{dataset_name}_correlation.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()
    return filename.name

def generate_markdown_report():
    """Generate the full markdown report."""
    report_path = OUTPUT_DIR / "data_description_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# C-MAPSS Data Description Report\n\n")
        f.write("This report provides a detailed statistical analysis of the raw C-MAPSS datasets (before normalization).\n\n")
        
        for name in DATASETS.keys():
            print(f"Processing {name}...")
            f.write(f"## Dataset: {name}\n\n")
            
            # Load raw data
            train_df, _, _ = load_raw_data(name)
            
            # Identify constant sensors to exclude from main stats
            constant_sensors = identify_constant_sensors(train_df)
            active_sensors = [s for s in SENSORS_TO_USE if s not in constant_sensors]
            
            f.write(f"**Total Samples**: {len(train_df)}\n\n")
            f.write(f"**Engines**: {train_df['unit_id'].nunique()}\n\n")
            
            if constant_sensors:
                f.write(f"**Constant Sensors (excluded)**: {', '.join(constant_sensors)}\n\n")
            
            # 1. Statistics Table
            f.write("### 1. Descriptive Statistics\n\n")
            stats = calculate_statistics(train_df, active_sensors)
            f.write(stats.to_markdown(floatfmt=".3f"))
            f.write("\n\n")
            
            # 2. Visualizations
            f.write("### 2. Visualizations\n\n")
            
            # Distributions
            dist_img = plot_distributions(train_df, name, active_sensors)
            f.write(f"#### Sensor Distributions\n")
            f.write(f"![Distributions](figures/{dist_img})\n\n")
            
            # Boxplots
            box_img = plot_boxplots(train_df, name, active_sensors)
            f.write(f"#### Outlier Analysis (Boxplots)\n")
            f.write(f"![Boxplots](figures/{box_img})\n\n")
            
            # Correlation
            corr_img = plot_correlation_heatmap(train_df, name, active_sensors)
            f.write(f"#### Correlation Heatmap\n")
            f.write(f"![Correlation](figures/{corr_img})\n\n")
            
            # 3. Insights
            f.write("### 3. Key Data Insights\n\n")
            insights = generate_insights(train_df, active_sensors)
            for insight in insights:
                f.write(f"- {insight}\n")
            f.write("\n")
            
            f.write("---\n\n")
            
    print(f"\nReport generated at: {report_path}")

def generate_insights(df, sensors):
    """Generate automatic textual insights based on data statistics."""
    insights = []
    
    # 1. Check for correlations
    corr_matrix = df[sensors].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    high_corr_pairs = []
    for column in upper.columns:
        for index in upper.index:
            if upper.loc[index, column] > 0.95:
                # Avoid duplicates (A,B) and (B,A) - handled by upper triangle
                 high_corr_pairs.append(f"{index} and {column}")
    
    if high_corr_pairs:
        top_pairs = high_corr_pairs[:3]
        insights.append(f"**High Correlation**: Strong linear relationship (>0.95) detected between: {', '.join(top_pairs)}. This suggests redundancy; one sensor in each pair could likely be dropped without information loss.")
        if len(high_corr_pairs) > 3:
            insights.append(f"*Note: {len(high_corr_pairs) - 3} other highly correlated pairs found.*")
            
    # 2. Check for variance
    variances = df[sensors].var()
    high_var_sensor = variances.idxmax()
    low_var_sensor = variances.idxmin()
    
    insights.append(f"**Variability**: `{high_var_sensor}` shows the highest variance ({variances.max():.2f}), indicating it fluctuates significantly across operating conditions. Conversely, `{low_var_sensor}` is the most stable.")

    # 3. Skewness (Distribution shape)
    skewness = df[sensors].skew()
    highly_skewed = skewness[abs(skewness) > 1.5]
    
    if not highly_skewed.empty:
        insights.append(f"**Distribution Shape**: Sensors like `{highly_skewed.index[0]}` show high skewness ({highly_skewed.iloc[0]:.2f}), indicating a non-normal distribution where extreme values are frequent.")
    else:
        insights.append(f"**Distribution Shape**: Most sensors follow a relatively symmetric distribution.")

    return insights

if __name__ == "__main__":
    generate_markdown_report()
