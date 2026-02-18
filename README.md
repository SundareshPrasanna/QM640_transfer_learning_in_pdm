# Transfer Learning for Fault Detection in Industrial Predictive Maintenance (QM640)

## 1. Project Overview
This capstone studies transfer learning for binary fault detection on NASA C-MAPSS turbofan data under domain shift.

- **Source domain**: `FD002`
- **Target domains**: `FD001`, `FD003`, `FD004`
- **Fault definition**: `fault_label = 1` if `RUL <= 20`, else `0`
- **Task**: Binary classification on sliding windows of multivariate sensor time series

The project evaluates direct transfer, standard fine-tuning, label efficiency, and **LSTM-only gradual unfreezing**.

## 2. Research Questions
- **RQ1 (Domain Shift)**: How much does performance degrade under direct transfer from source to targets?
- **RQ2 (Fine-Tuning)**: Does standard fine-tuning improve over direct transfer?
- **RQ3 (Robustness)**: How do RF, CNN, and LSTM compare under shift?
- **RQ4 (Label Efficiency)**: How much labeled target data is needed for meaningful gains?
- **RQ5 (Advanced Transfer)**: Does **LSTM gradual unfreezing** outperform standard LSTM fine-tuning?

## 3. Repository Structure
```text
synopsis_QM640/
├── data/
│   ├── raw/                       # Original C-MAPSS files
│   ├── processed/                 # Windowed/normalized arrays
│   └── labels/                    # Generated labels and metadata
├── models/saved/                  # Saved source-trained baselines
├── results/reports/               # CSV results, plots, run reports
├── scripts/                       # Experiment execution scripts
├── src/                           # Core pipeline and model code
├── QM640_Final_Report.md          # Final thesis report (source)
├── QM640_Final_Report_Submission.docx
├── requirements.txt
└── README.md
```

## 4. Environment Setup
### 4.1 Recommended environment
```bash
source ~/.pyenv/versions/[env_name]/bin/activate
```

### 4.2 Install dependencies
```bash
pip install -r requirements.txt
```

## 5. Reproducing the Latest Experiment Runs
Run in the order below.

### 5.1 Download data
```bash
python scripts/download_data.py
```

### 5.2 Preprocess data (RUL, labels, windows)
```bash
python -m src.preprocessing
```

### 5.3 Train source-domain baseline models
```bash
python scripts/train_baselines.py
```

### 5.4 Run all RQ experiments
```bash
python scripts/run_experiments.py --all
```

This executes:
- `python scripts/run_domain_shift.py` (RQ1)
- `python scripts/run_fine_tuning.py` (RQ2)
- `python scripts/run_robustness.py` (RQ3)
- `python scripts/run_label_efficiency.py` (RQ4)
- `python scripts/run_lstm_gradual_unfreezing.py` (RQ5)

## 6. Key Output Artifacts (Latest Runs)
Main outputs are under `results/reports/`:

- `rq1_results.csv`, `rq1_domain_shift_analysis.md`, `rq1_degradation.png`
- `rq2_results.csv`, `rq2_fine_tuning_analysis.md`, `rq2_comparison.png`
- `rq3_results.csv`, `rq3_robustness_analysis.md`
- `rq4_results.csv`, `rq4_label_efficiency_analysis.md`, `rq4_learning_curve.png`
- `lstm_gradual_unfreezing_results.csv`, `lstm_gradual_unfreezing_report.md`
- EDA plots: `eda_sensor_distributions.png`, `eda_correlation_heatmap.png`, `eda_outliers_boxplot.png`

## 7. Method Notes
- Engine-level splitting is used for transfer experiments to reduce leakage from overlapping windows.
- Effective inferential unit for transfer tests is target domain (`n=3`), so p-values are interpreted together with effect sizes.
- Random seed is fixed in config for reproducibility.

## 8. Dataset Citation
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for aircraft engine run-to-failure simulation. In *2008 International Conference on Prognostics and Health Management* (pp. 1-9). IEEE. https://doi.org/10.1109/PHM.2008.4711414

## 9. Author
Sundaresh Prasanna Chandran

Walsh College, QM640 Data Analytics Capstone
