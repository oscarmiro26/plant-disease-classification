import os
import torch
import numpy as np

# Data paths
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'color')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# Creating data directories in case we run it for the first time later after cloning (moving to Habrok)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data splitting
TEST_SPLIT_SIZE = 0.15
VALIDATION_SPLIT_SIZE = 0.15  # After the test split has been removed

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class labels mapping for easy access
ALL_CLASSES = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])
NUM_CLASSES = len(ALL_CLASSES)
LABEL_MAP = {label: i for i, label in enumerate(ALL_CLASSES)}
INV_LABEL_MAP = {i: label for label, i in LABEL_MAP.items()}
print(f"Found {NUM_CLASSES} classes in {RAW_DATA_DIR}")

print(f"Config Loaded:")
print(f"  Device: {DEVICE}")
print(f"  Number of Classes: {NUM_CLASSES}")
