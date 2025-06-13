import os
import torch
import numpy as np

# Data paths
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# Define potential data directories (for development and production in API server)
DEV_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "color")
PROD_DATA_PATH = os.path.join(PROJECT_ROOT, "samples")

try:
    # Attempt to list the contents of the dev directory.
    # This will raise FileNotFoundError if DEV_DATA_PATH doesn't exist or isn't a directory.
    if not os.path.isdir(DEV_DATA_PATH):
        raise FileNotFoundError  # Explicitly raise if it's not a directory
    os.listdir(DEV_DATA_PATH)  # Access the path to trigger error if it doesn't exist
    RAW_DATA_DIR = DEV_DATA_PATH
    print(f"Using development data directory: {RAW_DATA_DIR}")
except FileNotFoundError:
    # Fallback to production/samples directory
    RAW_DATA_DIR = PROD_DATA_PATH
    print(
        f"Development data directory '{DEV_DATA_PATH}' not found or not accessible. Using production/fallback data directory: {RAW_DATA_DIR}"
    )
    # Optionally, add a check here if PROD_DATA_PATH must also exist and be a directory
    if not os.path.isdir(RAW_DATA_DIR):
        # This is a more critical error if neither path is valid.
        # The script will likely fail later at os.listdir() for ALL_CLASSES anyway.
        print(
            f"WARNING: Fallback data directory {RAW_DATA_DIR} also not found or not a directory."
        )
        # You might want to raise an error here if this is an unrecoverable state:
        # raise FileNotFoundError(f"Neither development path '{DEV_DATA_PATH}' nor fallback path '{PROD_DATA_PATH}' are valid directories.")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Creating data directories in case we run it for the first time later after cloning (moving to Habrok)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data splitting
TEST_SPLIT_SIZE = 0.15
VALIDATION_SPLIT_SIZE = 0.15

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels mapping for easy access
ALL_CLASSES = sorted(
    [
        d
        for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ]
)
NUM_CLASSES = len(ALL_CLASSES)
LABEL_MAP = {label: i for i, label in enumerate(ALL_CLASSES)}
INV_LABEL_MAP = {i: label for label, i in LABEL_MAP.items()}
print(f"Found {NUM_CLASSES} classes in {RAW_DATA_DIR}")

print(f"Config Loaded:")
print(f"Device: {DEVICE}")
print(f"Number of Classes: {NUM_CLASSES}")
