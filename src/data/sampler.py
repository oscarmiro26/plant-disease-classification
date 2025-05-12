import torch
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

def calculate_class_weights(dataframe: pd.DataFrame, label_map: dict):
    """
    Calculates the class weights inversely proportional to class frequency, scaled by num_classes.
    """
    print("Calculating class weights for loss function...")
    class_counts = dataframe['label'].value_counts().sort_index()
    num_classes = len(label_map)
    total_samples = len(dataframe)

    class_weights = total_samples / (num_classes * class_counts)

    weights_tensor = torch.zeros(num_classes, dtype=torch.float32)
    for label, index in label_map.items():
        if label in class_weights.index:
            weights_tensor[index] = class_weights[label]
        else:
            print(f"Warning (calculate_class_weights): Class '{label}' not found in this DataFrame split. Assigning loss weight 1.0.")
            weights_tensor[index] = 1.0

    print(f"Class weights calculation complete.")
    return weights_tensor


def create_sampler(dataframe: pd.DataFrame, label_map: dict):
    """
    Creates a WeightedRandomSampler using inverse frequency weighting for samples.
    """
    print("Creating WeightedRandomSampler...")

    class_counts = dataframe['label'].value_counts()
    num_samples = len(dataframe)

    class_weights_map = {label: num_samples / count for label, count in class_counts.items()}

    try:
        sample_weights = dataframe['label'].map(class_weights_map).to_numpy()
    except Exception as e:
         print(f"Error mapping sampler weights: {e}. Check if all labels in dataframe exist in class_counts.")
         sample_weights = np.ones(len(dataframe))


    # Added this from the internet to fix errors with NaN values
    if np.isnan(sample_weights).any():
        print("Warning: NaN sample weights detected. This might occur if labels in the DataFrame are missing from class_counts.")
        median_weight = np.nanmedian(sample_weights) if not np.all(np.isnan(sample_weights)) else 1.0
        sample_weights = np.nan_to_num(sample_weights, nan=median_weight)
        print(f"Replaced NaN sample weights with: {median_weight:.2f}")


    sample_weights_tensor = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    print(f"WeightedRandomSampler created.")
    return sampler
    