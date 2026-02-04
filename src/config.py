"""
Configuration file for Transfer Learning Fault Detection project.
All hyperparameters and constants are defined here.
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# C-MAPSS dataset files
DATASETS = {
    "FD001": {"train": "train_FD001.txt", "test": "test_FD001.txt", "rul": "RUL_FD001.txt"},
    "FD002": {"train": "train_FD002.txt", "test": "test_FD002.txt", "rul": "RUL_FD002.txt"},
    "FD003": {"train": "train_FD003.txt", "test": "test_FD003.txt", "rul": "RUL_FD003.txt"},
    "FD004": {"train": "train_FD004.txt", "test": "test_FD004.txt", "rul": "RUL_FD004.txt"},
}

# Column names for raw data
COLUMN_NAMES = (
    ["unit_id", "cycle"] +
    [f"op_setting_{i}" for i in range(1, 4)] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# Source and target domains
SOURCE_DOMAIN = "FD002"
TARGET_DOMAINS = ["FD001", "FD003", "FD004"]

# ============================================================================
# PREPROCESSING
# ============================================================================
# Threshold to label as fault (RUL <= FAULT_THRESHOLD)
FAULT_THRESHOLD = 20

# Sliding window size for CNN/LSTM
WINDOW_SIZE = 30

# Sensors to use (some sensors have constant values and are dropped)
# Will be determined during EDA - initially use all
SENSORS_TO_USE = [f"sensor_{i}" for i in range(1, 22)]

# Operating settings
OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
# Random Forest
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

# CNN
CNN_PARAMS = {
    "input_channels": 21,  # Number of sensors
    "conv_filters": [64, 128, 128],
    "kernel_size": 3,
    "fc_units": 64,
    "dropout": 0.3,
}

# LSTM
LSTM_PARAMS = {
    "input_size": 21,  # Number of sensors
    "hidden_size": 64,
    "num_layers": 2,
    "bidirectional": True,
    "fc_units": 64,
    "dropout": 0.3,
}

# ============================================================================
# TRAINING
# ============================================================================
TRAINING_PARAMS = {
    "batch_size": 64,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 10,
}

# Fine-tuning
FINE_TUNING_PARAMS = {
    "epochs": 20,
    "learning_rate": 1e-4,  # Lower LR for fine-tuning
    "freeze_layers": True,  # Freeze base layers, train head only
}

# Label efficiency experiments
LABEL_PERCENTAGES = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20%

# ============================================================================
# EVALUATION
# ============================================================================
# Statistical significance level
ALPHA = 0.05

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# DEVICE
# ============================================================================
import torch

def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
