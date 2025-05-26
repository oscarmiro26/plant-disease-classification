from typing import Callable, Tuple
import numpy as np
import cv2
from skimage import color, measure, util
from skimage.filters import gabor, threshold_otsu # Added gaussian
from skimage.feature import (
    hog, 
    local_binary_pattern, 
    graycomatrix, 
    graycoprops,
)
from joblib import Parallel, delayed

from .datasets import PlantDiseaseDataset # Assuming PlantDiseaseDataset is in datasets.py in the same directory


def segment_leaf(image: np.ndarray) -> np.ndarray:
    """
    Leaf segmentation using HSV pre-mask + GrabCut.

    Args:
        image: H×W×3 uint8 RGB image as a NumPy array.
    Returns:
        mask: H×W uint8 binary mask where leaf pixels are 255 and background is 0.
    """
    # === 1) Rough pre-mask via HSV threshold ===
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([25, 40, 40])
    upper = np.array([100, 255, 255])
    rough = cv2.inRange(hsv, lower, upper)

    # === 2) Initialize GrabCut mask ===
    #  - 0,2 = background; 1,3 = foreground
    gc_mask = np.where(rough>0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    #  Define a rectangle inset by 5% on each side (for safety)
    h, w = image.shape[:2]
    pad_h, pad_w = int(0.05*h), int(0.05*w)
    rect = (pad_w, pad_h, w-2*pad_w, h-2*pad_h)

    # === 3) Run GrabCut ===
    cv2.grabCut(image, gc_mask, rect, bgdModel, fgdModel, 
                iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

    # === 4) Build final mask ===
    mask = np.where(
        (gc_mask==cv2.GC_FGD) | (gc_mask==cv2.GC_PR_FGD), 
        255, 
        0
    ).astype('uint8')

    # === 5) Morphological cleanup ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # === 6) Final thresholding to remove noise ===
    thresh = threshold_otsu(mask)
    mask = (mask > thresh).astype('uint8') * 255
    # === 7) Segment the leaf from the original image ===
    segmented_leaf = cv2.bitwise_and(image, image, mask=mask)

    return segmented_leaf

def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Feature extraction method: Compute Gabor filter and HSV histogram features.
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
