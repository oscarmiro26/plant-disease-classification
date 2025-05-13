import torch
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

def _inverse_frequency_map(dataframe: pd.DataFrame, label_map: dict) -> dict:
    """
    Computes inverse frequency weight for each class label based on sample counts.
    Returns a mapping from label to inverse frequency weight.
    """
    # Count samples per class in the dataframe
    class_counts = dataframe['label'].value_counts().to_dict()
    total_samples = len(dataframe)
    
    # Inverse frequency: total_samples / class_count
    inv_freq = {label: total_samples / count for label, count in class_counts.items() if count > 0}
    
    # Ensure every label in label_map has an entry (default weight 1.0)
    for label in label_map:
        inv_freq.setdefault(label, 1.0)
    
    return inv_freq


def calculate_class_weights(dataframe: pd.DataFrame, label_map: dict) -> torch.Tensor:
    """
    Calculates class weights tensor for loss, inversely proportional to class frequency
    and normalized by the number of classes.
    """
    print("Calculating class weights for loss function...")
    num_classes = len(label_map)
    inv_freq_map = _inverse_frequency_map(dataframe, label_map)

    # Build weights tensor where index = class index
    weights = [inv_freq_map[label] / num_classes for label in label_map]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("Class weights calculation complete.")
    return weights_tensor


def create_sampler(
    dataframe: pd.DataFrame, 
    label_map: dict, 
    use_mixing_sampler: bool = False, 
    alpha: float = 0.5
) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler to address class imbalance by sampling
    inversely proportional to class frequency.
    If use_mixing_sampler is True, a mixing sampler is created where the per-sample
    weight is a mix of inverse frequency and uniform weights: w = \alpha * w_inv + (1 - \alpha) * 1.
    """
    print("Creating WeightedRandomSampler...")
    # Compute inverse frequency map
    inv_freq_map = _inverse_frequency_map(dataframe, label_map)

    # Map each sample's label to its weight
    sample_weights = dataframe['label'].map(inv_freq_map).to_numpy(dtype=np.float64)

    if use_mixing_sampler:
        sample_weights = alpha * sample_weights + (1 - alpha)

    # Handle any unexpected NaNs by replacing with median weight
    if np.isnan(sample_weights).any():
        median_weight = np.nanmedian(sample_weights)
        sample_weights = np.nan_to_num(sample_weights, nan=median_weight)
        print(f"Warning: NaN sample weights detected. Replaced with median weight {median_weight:.2f}.")

    # Create tensor of sample weights
    sample_weights_tensor = torch.from_numpy(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )
    print("WeightedRandomSampler created.")
    return sampler
