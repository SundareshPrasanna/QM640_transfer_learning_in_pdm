# Transfer Learning for Fault Detection in Industrial Predictive Maintenance

## Overview
Binary fault detection using transfer learning on NASA C-MAPSS turbofan engine dataset.

- **Source Domain**: FD002 (260 engines, 6 operating conditions)
- **Target Domains**: FD001, FD003, FD004
- **Fault Definition**: RUL ≤ 20 cycles

## Project Structure
```
synopsis_QM640/
├── data/raw/              # Original C-MAPSS files
├── data/processed/        # Preprocessed datasets
├── data/labels/           # Generated fault labels
├── notebooks/             # EDA and analysis notebooks
├── models/saved/          # Trained model checkpoints
├── src/                   # Source code
│   ├── models/            # Model architectures
│   ├── config.py          # Hyperparameters
│   ├── preprocessing.py   # Data processing
│   └── ...
├── results/               # Figures and reports
├── scripts/               # Execution scripts
└── requirements.txt
```

## Quick Start
```bash
# Activate environment
source ~/.pyenv/versions/[YOUR_PYENV_NAME]/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py

# Run experiments
python scripts/run_experiments.py --all
```

## Research Hypotheses

| Research Question | Null Hypothesis ($H_0$) | Alternative Hypothesis ($H_1$) |
|-------------------|-------------------------|--------------------------------|
| **RQ1: Domain Shift** | No significant performance difference between source and target domains (direct transfer). | Significant performance degradation under direct transfer. |
| **RQ2: Fine-Tuning** | Fine-tuning does not improve performance over direct transfer. | Fine-tuning significantly improves performance. |
| **RQ3: Robustness** | All architectures (RF, CNN, LSTM) perform equally under domain shift. | At least one architecture is significantly more robust. |

## Models Experimented

### 1. Random Forest (Baseline)
- **Role**: Traditional ML baseline for comparison.
- **Features**: Hand-crafted statistical features (mean, std, min, max, slope, percentiles) extracted from 30-cycle sliding windows.
- **Config**: 200 estimators, Max depth 20, Balanced class weights.

### 2. Convolutional Neural Network (CNN)
- **Role**: Feature extractor for local temporal patterns.
- **Architecture**: 3-layer 1D-Conv stack + Global Average Pooling + Classification Head.
- **Input**: Raw multivariate time-series (window size 30).
- **Transfer Strategy**: Frozen convolutional layers, fine-tuned classification head.

### 3. Long Short-Term Memory (LSTM)
- **Role**: Sequence model for long-term dependencies (Reserved for Final Report).
- **Architecture**: Bidirectional LSTM layers.

## Author
Sundaresh Prasanna Chandran - Walsh College QM640 Capstone
