from typing import Callable, Tuple
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,  
    RFE,
    f_classif,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from skimage import color, measure, util
from skimage.filters import gabor
from skimage.feature import (
    hog, 
    local_binary_pattern, 
    graycomatrix, 
    graycoprops,
)
import joblib
from joblib import Parallel, delayed
import torchvision.transforms as T

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config
from ..utils.logger import setup_logger


N_JOBS = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

PARAM_GRID = {
    'svc__C': [0.001, 0.01, 0.1, 1, 10],
    'svc__gamma': ['scale', 'auto', 0.001, 0.0001],
}

class CorrelationFilter(BaseEstimator):
    """
    Drop features that have pairwise |corr| > threshold.
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Compute absolute correlation matrix
        corr = np.abs(np.corrcoef(X, rowvar=False))
        # Find upper-triangle indices where corr > threshold
        upper = np.triu_indices_from(corr, k=1)
        to_drop = set()
        for i, j in zip(*upper):
            if corr[i, j] > self.threshold:
                to_drop.add(j)
        self.keep_idx_ = [i for i in range(X.shape[1]) if i not in to_drop]
        return self

    def transform(self, X):
        return X[:, self.keep_idx_]
    

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
    n_jobs: int = N_JOBS,
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
        arr = img.numpy().astype(np.float32)
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

def train_and_evaluate():
    """
    Main entrypoint for SVM training:
      - Splits data
      - Extracts features
      - Performs GridSearchCV
      - Trains final model
      - Evaluates on test set
      - Saves model, metrics, and confusion matrix
    """
    logger, RUN_DIR = setup_logger(
        model_name="svm",
        base_log_dir=config.LOG_DIR
    )
    logger.info(f"Starting SVM training and evaluation run in {RUN_DIR}")
    logger.info(f"Parameter grid: {PARAM_GRID}")

    # Data splits
    # We'll concatenate val_df with train_df for training w/ k-fold and test_df for final evaluation.
    logger.info("Creating data splits...")
    train_df, val_df, test_df = create_splits(
        data_dir=config.RAW_DATA_DIR,
        label_map=config.LABEL_MAP,
        test_size=config.TEST_SPLIT_SIZE,
        val_size=config.VALIDATION_SPLIT_SIZE,
    )
    train_df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info("Creating datasets...")

    data_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    train_ds = PlantDiseaseDataset(train_df, transform=data_transform)
    test_ds  = PlantDiseaseDataset(test_df,  transform=data_transform)

    # feature_extraction_fn = extract_features_1
    # feature_extraction_fn.__name__ = "HOG_LBP" # Set the name for logging
    feature_extraction_fn = extract_features_2
    feature_extraction_fn.__name__ = "GLCM_Shape_HSV" # Set the name for logging
    # feature_extraction_fn = extract_features_3
    # feature_extraction_fn.__name__ = "Gabor_HSV" # Set the name for logging

    # Load and prepare data for SVM
    logger.info("Loading and preparing training data for SVM using feature extraction: %s", feature_extraction_fn.__name__)
    X_train, y_train = load_and_prepare_data(train_ds, feature_extraction_fn)
    logger.info(f"Training data shape: Features {X_train.shape}, Labels {y_train.shape}")

    logger.info("Loading and preparing test data for SVM...")
    X_test, y_test = load_and_prepare_data(test_ds, feature_extraction_fn)
    logger.info(f"Test data shape: Features {X_test.shape}, Labels {y_test.shape}")

    logger.info("Data preparation complete. Commencing SVM training...")

    # # Pipeline for feature extraction #1
    # pipeline = Pipeline([
    #     ('var_thresh',      VarianceThreshold(threshold=1e-3)),
    #     ('corr_filter',     CorrelationFilter(threshold=0.9)),
    #     ('univariate',      SelectKBest(f_classif, k=100)), # SelectKBest
    #     ('scaler',          StandardScaler()),
    #     ('rfe',             RFE(estimator=SVC(kernel="linear", 
    #                                         max_iter=5000), n_features_to_select=75, step=0.1)),
    #     ('pca',             PCA(n_components=0.95, whiten=True)),
    #     ('svc',             SVC(class_weight='balanced', 
    #                             cache_size=2000, kernel='rbf'))
    # ])

    # Pipeline for feature extraction #2 and #3
    pipeline = Pipeline([
        ('var_thresh',      VarianceThreshold(threshold=1e-3)),
        ('corr_filter',     CorrelationFilter(threshold=0.9)),
        ('scaler',          StandardScaler()),        
        ('pca',             PCA(n_components=0.95, whiten=True)),
        ('svc',             SVC(class_weight='balanced', 
                                cache_size=2000, kernel='rbf'))
    ])

    # Hyperparameter tuning
    logger.info("Starting grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=5,
        n_jobs=N_JOBS,
        verbose=3,
        return_train_score=True,
        pre_dispatch=f"{N_JOBS * 2}*n_jobs"   # throttle the number of launched jobs
    )
    
    # Fit the model using GridSearchCV using tqdm_joblib for progress bar
    n_candidates = len(list(ParameterGrid(PARAM_GRID)))
    total_fits   = n_candidates * grid_search.cv
    with tqdm_joblib(tqdm(desc="GridSearchCV", total=total_fits)):
        grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logger.info("Grid search complete.")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    best_c = best_params['svc__C']
    best_gamma = best_params['svc__gamma']

    # Training
    logger.info("Training SVM model...")
    best_model.fit(X_train, y_train)
    logger.info("SVM training complete.")

    # Save the trained model
    model_name = f'svm_model_{feature_extraction_fn.__name__}_{best_c}_{best_gamma}'
    model_filename = f'{model_name}.pkl'
    model_path = os.path.join(RUN_DIR, model_filename)
    joblib.dump(best_model, model_path)
    logger.info(f"SVM model saved to {model_path}")

    # Test evaluation
    logger.info("Evaluating SVM model on the test set...")
    y_pred = best_model.predict(X_test)

    # Ensure labels for report are correctly mapped
    target_names = [config.INV_LABEL_MAP[i] for i in sorted(config.INV_LABEL_MAP.keys()) if i < config.NUM_CLASSES]
    labels_for_report = list(range(config.NUM_CLASSES))

    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred, labels=labels_for_report)
    report = classification_report(
        y_test, y_pred,
        labels=labels_for_report,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    logger.info("Confusion Matrix:\n%s", cm)
    logger.info("Classification Report:\n%s", report)

    # Classification report to CSV
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test, y_pred,
        labels=labels_for_report,
        zero_division=0
    )

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'class': target_names,
        'precision': prec,
        'recall': rec,
        'f1-score': f1,
        'support': sup
    })
    metrics_file_name = f'{model_name}_classification_metrics.csv'
    metrics_path = os.path.join(RUN_DIR, metrics_file_name)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        cm,
        index=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)],
        columns=[config.INV_LABEL_MAP[i] for i in range(config.NUM_CLASSES)]
    )
    cm_file_name = f'{model_name}_confusion_matrix.csv'
    cm_csv = os.path.join(RUN_DIR, cm_file_name)
    cm_df.to_csv(cm_csv)
    logger.info(f"Confusion matrix saved to {cm_csv}")
    logger.info("SVM run finished.")

    # ### DEBUGGING (comment out above code) ###
    # train_df, val_df, test_df = create_splits(
    #     data_dir=config.RAW_DATA_DIR,
    #     label_map=config.LABEL_MAP,
    #     test_size=config.TEST_SPLIT_SIZE,
    #     val_size=config.VALIDATION_SPLIT_SIZE,
    # )
    # # Data Transform w/ feature extraction
    # data_transform = T.Compose([
    #     T.Resize((256, 256)),
    #     T.ToTensor(),
    # ])

    # # # Data transform w/o feature extraction
    # # data_transform = T.Compose([
    # #     T.Resize((64, 64)),
    # #     T.ToTensor(),
    # # ])

    # # Subset for quick debugging
    # train_df = train_df.sample(n=100, random_state=42).reset_index(drop=True)
    # test_df  = test_df.sample(n=100, random_state=42).reset_index(drop=True)

    # train_ds = PlantDiseaseDataset(train_df, transform=data_transform)
    # test_ds  = PlantDiseaseDataset(test_df,  transform=data_transform)

    # feature_extraction_fn = extract_features_1

    # X_train, y_train = load_and_prepare_data(train_ds, feature_extraction_fn)
    # X_test, y_test = load_and_prepare_data(test_ds, feature_extraction_fn)

    # # Scaler + SVM pipeline
    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),               
    #     ('svc', SVC(class_weight='balanced', cache_size=5000, kernel='rbf'))
    # ])

    # # Hyperparameter tuning
    # grid_search = GridSearchCV(
    #     pipeline,
    #     PARAM_GRID,
    #     cv=5,
    #     n_jobs=N_JOBS,
    #     verbose=3,
    #     return_train_score=True,
    #     pre_dispatch=f"{N_JOBS * 2}*n_jobs"   # throttle the number of launched jobs
    # )

    # grid_search.fit(X_train, y_train)

    # ### DEBUGGING (comment out above code) ###

if __name__ == "__main__":
    train_and_evaluate()