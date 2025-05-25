\
# filepath: /scratch/s5142822/plant-disease-classification/src/data/svm_preprocessing.py
from typing import Callable, Tuple
import numpy as np
from skimage import color, measure, util
from skimage.filters import gabor
from skimage.feature import (
    hog, 
    local_binary_pattern, 
    graycomatrix, 
    graycoprops,
)
from joblib import Parallel, delayed

from .datasets import PlantDiseaseDataset # Assuming PlantDiseaseDataset is in datasets.py in the same directory


def extract_features_1(img: np.ndarray) -> np.ndarray:
    """
    Feature extraction #1: compute HOG and LBP descriptors from a single image.
    Feature dimensions:
    - HOG: 2048 features (8 orientations × 16 × 16)
    - LBP: 10 features (10 bins)
    Total: 2058 features.
    Args:
        img (np.ndarray): Input image array. Can be 2D (H×W) or
                               3D (C×H×W) for RGB data.
    Returns:
        np.ndarray: Concatenated 1D feature vector [hog_feats | lbp_hist].
    """
    # Convert to grayscale
    gray_img = color.rgb2gray(img)

    # HOG on 2D gray image
    hog_feats = hog(gray_img, orientations=8,
                    pixels_per_cell=(16,16),
                    cells_per_block=(1,1),
                    block_norm='L2',
                    visualize=False,
                    feature_vector=True)

    # LBP on uint8
    lbp_image = (gray_img * 255).astype(np.uint8)
    lbp = local_binary_pattern(lbp_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    return np.concatenate([hog_feats, lbp_hist])

def extract_features_2(img: np.ndarray) -> np.ndarray:
    """
    Feature extraction #2: compute HSV histograms, GLCM texture stats,
    and shape descriptors from a single RGB image.
    Feature dimensions:
    - Color (HSV): 48 features (16 bins × 3 channels)
    - Texture (GLCM): 16 features (4 angles × 4 stats)
    - Shape: 11 features (lesion count, mean area, std area, mean perimeter,
      area, perimeter, eccentricity, solidity, extent, major axis length,
      aspect ratio)
    Total: 75 features.
    Args:
        image_np (np.ndarray): Input RGB image array of shape (256, 256, 3).

    Returns:
        np.ndarray: Concatenated feature vector [color | texture | shape].
    """
    # Ensure float in [0,1]
    image = util.img_as_float(img)

    # Color (HSV)
    hsv = color.rgb2hsv(image, channel_axis=2)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 1), density=True)
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=16, range=(0, 1), density=True)
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=16, range=(0, 1), density=True)
    color_features = np.concatenate([h_hist, s_hist, v_hist])

    # Texture (GLCM)
    gray = color.rgb2gray(image)
    gray_u8 = util.img_as_ubyte(gray)
    glcm = graycomatrix(
        gray_u8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "homogeneity", "energy", "correlation"]
    texture_features = np.hstack([graycoprops(glcm, p).flatten() for p in props])

    # Shape descriptors
    binary = gray_u8 > 0
    regions = measure.regionprops(measure.label(binary))
    # Lesion count & size statistics
    lesion_count = len(regions)
    if lesion_count > 0:
        areas     = np.array([r.area for r in regions], dtype=np.float32)
        perims    = np.array([r.perimeter for r in regions], dtype=np.float32)
        mean_area = areas.mean()
        std_area  = areas.std()
        mean_perim= perims.mean()
    else:
        mean_area = std_area = mean_perim = 0.0

    # Shape features from largest region
    if regions:
        r         = max(regions, key=lambda reg: reg.area)
        area      = float(r.area)
        perimeter = float(r.perimeter)
        ecc       = float(r.eccentricity)
        solidity  = float(r.solidity)
        extent    = float(r.extent)
        maj       = float(r.major_axis_length)
        min_      = float(r.minor_axis_length or 1.0)
        aspect    = maj / min_
    else:
        area = perimeter = ecc = solidity = extent = maj = aspect = 0.0

    shape_features = np.array([
        lesion_count, mean_area, std_area, mean_perim,
        area, perimeter, ecc, solidity, extent, maj, aspect
    ], dtype=np.float32)

    # Concatenate all features
    features = np.concatenate([color_features, texture_features, shape_features], axis=0)

    return features

def extract_features_3(img: np.ndarray) -> np.ndarray:
    """
    Feature extraction #3: Compute Gabor filter and HSV histogram features.
    Feature dimensions:
    - Gabor: 32 features (4 frequencies × 4 orientations × 2 stats)
    - HSV: 48 features (16 bins × 3 channels)
    Total: 80 features.
    Args:
        image_np (np.ndarray): Input RGB image array of shape (256, 256, 3).
    Returns:
        np.ndarray: Concatenated feature vector [gabor | hsv].
    """
    # Convert image to float and grayscale for Gabor filter
    gray = color.rgb2gray(util.img_as_float(img))
    
    # Gabor filter bank (4 frequencies × 4 orientations)
    gabor_feats = []
    frequencies = [0.1, 0.2, 0.3, 0.4]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for f in frequencies:
        for theta in thetas:
            filt_real, _ = gabor(gray, frequency=f, theta=theta)
            gabor_feats.append(filt_real.mean())
            gabor_feats.append(filt_real.std())
    gabor_feats = np.array(gabor_feats)
    
    # HSV color histogram (16 bins per channel)
    hsv = color.rgb2hsv(img)
    h_hist, _ = np.histogram(hsv[..., 0], bins=16, range=(0,1), density=True)
    s_hist, _ = np.histogram(hsv[..., 1], bins=16, range=(0,1), density=True)
    v_hist, _ = np.histogram(hsv[..., 2], bins=16, range=(0,1), density=True)
    color_feats = np.concatenate([h_hist, s_hist, v_hist])
    
    # Concatenate all features
    return np.concatenate([gabor_feats, color_feats])


def load_and_prepare_data(
    dataset: PlantDiseaseDataset,
    feature_extraction_fn: Callable[[np.ndarray], np.ndarray],
    n_jobs: int # Assuming N_JOBS will be passed or defined elsewhere
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel loading and feature extraction for an entire dataset.

    Args:
        dataset (PlantDiseaseDataset): PyTorch-like dataset yielding (img, label).
        feature_extraction_fn (Callable): Function to compute features for one image.
        n_jobs (int): Number of parallel workers.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) where X is (N, D) feature array,
                                       y is (N,) label array.
    """
    # helper to process one sample
    def _process(idx_img_lbl):
        img, lbl = idx_img_lbl
        if img.ndim == 3:
            # H×W×C
            img = np.transpose(img, (1, 2, 0))
        # Ensure img is a NumPy array before calling .numpy()
        # If it's already a NumPy array, this won't do anything.
        # If it's a Tensor, it will convert it.
        # However, the type hint says np.ndarray, so it should already be one.
        # The original code had img.numpy().astype(np.float32)
        # This implies img was a tensor at that point.
        # Let's assume the input `img` to feature_extraction_fn is always np.ndarray
        # as per its type hint. The conversion from tensor should happen before calling this.
        # The _process function receives `img` from the dataset, which might be a Tensor.
        
        # If PlantDiseaseDataset returns tensors:
        if hasattr(img, 'numpy'):
            arr = img.numpy().astype(np.float32)
        else: # Assuming it's already a numpy array if not a tensor
            arr = img.astype(np.float32)

        feats = feature_extraction_fn(arr)
        return feats, lbl
    
    # map over all samples in parallel
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=1)(
        delayed(_process)(sample) for sample in dataset
    )

    all_features, all_labels = zip(*results)
    X = np.stack(all_features, axis=0)
    y = np.array(all_labels)
    return X, y
