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
source ~/.pyenv/versions/nabel_agents_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py

# Run experiments
python scripts/run_experiments.py --all
```

## Research Questions
1. **RQ1**: Domain shift impact on fault detection
2. **RQ2**: Fine-tuning effectiveness
3. **RQ3**: Model architecture robustness (RF vs CNN vs LSTM)
4. **RQ4**: Label efficiency (minimum labeled data required)

## Author
Sundaresh Prasanna Chandran - Walsh College QM640 Capstone
