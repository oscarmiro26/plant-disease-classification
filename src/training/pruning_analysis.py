import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from torchvision.models import resnet50

from .config import MODELS_DIR, PLOTS_DIR, DEVICE, INV_LABEL_MAP


def analyze_filter_norms(model):
    layer_norms = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data
            # Compute L1 norms per filter
            norms = weights.view(weights.size(0), -1).abs().sum(dim=1).cpu().numpy()
            layer_norms[name] = {
                'mean': np.mean(norms),
                'std': np.std(norms),
                'min': np.min(norms),
                'max': np.max(norms),
                'percentile_5': np.percentile(norms, 5)
            }
    return layer_norms

def visualize_filter_norms(norms_data):
    # Visualize
    plt.figure(figsize=(12, 6))
    for i, (layer, stats) in enumerate(norms_data.items()):
        plt.bar(i, stats['mean'], yerr=stats['std'], alpha=0.7)
    plt.axhline(y=np.mean([s['percentile_5'] for s in norms_data.values()]), 
                color='r', linestyle='--', label='5th Percentile Average')
    plt.xticks(range(len(norms_data)), list(norms_data.keys()), rotation=90)
    plt.title('ResNet50 Filter Norm Distribution')
    plt.ylabel('L1 Norm')
    plt.legend()
    plt.show()

    file_path = os.path.join(PLOTS_DIR, "filter_norms.png")
    plt.savefig(file_path)
    print(f"Filter norms visualization saved to {file_path}")

def geometric_median(filters):
    """Compute FPGM's approximated geometric median"""
    flat_filters = filters.reshape(filters.shape[0], -1)
    pairwise_dist = cdist(flat_filters, flat_filters, 'euclidean')
    sum_dist = pairwise_dist.sum(axis=1)
    return flat_filters[np.argmin(sum_dist)]

def fpgm_distance_analysis(model):
    """FPGM-specific layer analysis"""
    results = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weights = module.weight.data.cpu().numpy()
            n_filters = weights.shape[0]
            
            # Compute geometric median
            gm = geometric_median(weights)
            
            # Calculate Euclidean distances to GM
            flat_weights = weights.reshape(n_filters, -1)
            distances = np.linalg.norm(flat_weights - gm, axis=1)
            
            # Compute redundancy metric
            sorted_dist = np.sort(distances)
            redundancy_score = np.mean(sorted_dist[:int(0.2*n_filters)]) / np.mean(distances)
            
            results[name] = {
                'distances': distances,
                'gm': gm,
                'redundancy_score': redundancy_score,
                'mean_distance': np.mean(distances),
                'min_distance': np.min(distances),
                'percentile_10': np.percentile(distances, 10)
            }
    return results

def visualize_fpgm_data(fpgm_data):
    # Visualization
    plt.figure(figsize=(15, 8))
    for i, (layer, stats) in enumerate(fpgm_data.items()):
        plt.subplot(2, 1, 1)
        plt.bar(i, stats['redundancy_score'], color='skyblue')
        
        plt.subplot(2, 1, 2)
        plt.scatter([i]*len(stats['distances']), stats['distances'], 
                    alpha=0.4, c='purple', s=10)
        plt.errorbar(i, stats['mean_distance'], yerr=stats['mean_distance']/3, 
                    fmt='o', c='red', label='Mean±SD' if i==0 else "")
        
    plt.subplot(2, 1, 1)
    plt.title('FPGM Redundancy Score by Layer')
    plt.ylabel('Redundancy Score\n(lower = more redundant)')
    plt.axhline(y=0.3, color='r', linestyle='--', label='High Redundancy Threshold')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Filter-to-Geometric Median Distances')
    plt.ylabel('Euclidean Distance')
    plt.axhline(y=np.percentile([d['percentile_10'] for d in fpgm_data.values()], 50), 
                color='g', linestyle='--', label='Median P10 Distance')
    plt.legend()

    plt.tight_layout()
    plt.show()

    file_path = os.path.join(PLOTS_DIR, "fpgm_analysis.png")
    plt.savefig(file_path)
    print(f"FPGM analysis visualization saved to {file_path}")


if __name__ == "__main__":
    # Load model
    base = resnet50(pretrained=False)
    num_classes = len(INV_LABEL_MAP)
    num_ftrs = base.fc.in_features
    base.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(512, num_classes)
    )
    resnet_model_path = os.path.join(MODELS_DIR, "resnet50_9897.pth")
    state = torch.load(resnet_model_path, map_location="cpu")
    base.load_state_dict(state)
    model = base.eval()
    model.to(DEVICE)
    print("Model loaded and ready for analysis.")

    # Analyze filter norms
    print("Starting filter norms analysis...")
    norms_data = analyze_filter_norms(model)
    visualize_filter_norms(norms_data)
    # FPGM distance analysis
    print("Starting FPGM distance analysis...")
    fpgm_data = fpgm_distance_analysis(model)
    visualize_fpgm_data(fpgm_data)
    print("Analysis complete.")
