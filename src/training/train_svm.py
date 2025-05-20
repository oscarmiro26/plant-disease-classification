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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from skimage import color, measure, util
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
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

def extract_features_1(image_np: np.ndarray) -> np.ndarray:
    """
    Feature extraction #1: compute HOG and LBP descriptors from a single image.

    Args:
        image_np (np.ndarray): Input image array. Can be 2D (H×W) or
                               3D (C×H×W) for RGB data.

    Returns:
        np.ndarray: Concatenated 1D feature vector [hog_feats | lbp_hist].
    """
    if image_np.ndim == 3:
        # H×W×C
        img = np.transpose(image_np, (1, 2, 0))
        # skimage rgb2gray returns float image in [0, 1]
        gray_img = color.rgb2gray(img)
    else:
        # already H×W
        gray_img = image_np

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

def extract_features_2(image_np: np.ndarray) -> np.ndarray:
    """
    Feature extraction #2: compute HSV histograms, GLCM texture stats,
    and shape descriptors from a single RGB image.

    Args:
        image_np (np.ndarray): Input RGB image array of shape (256, 256, 3).

    Returns:
        np.ndarray: Concatenated feature vector [color | texture | shape].
    """
    # Ensure float in [0,1]
    image = util.img_as_float(image_np)

    # 1. Color (HSV)
    hsv = color.rgb2hsv(image, channel_axis=2)
    h_hist, _ = np.histogram(hsv[:, :, 0], bins=16, range=(0, 1), density=True)
    s_hist, _ = np.histogram(hsv[:, :, 1], bins=16, range=(0, 1), density=True)
    v_hist, _ = np.histogram(hsv[:, :, 2], bins=16, range=(0, 1), density=True)
    color_features = np.concatenate([h_hist, s_hist, v_hist])

    # 2. Texture (GLCM)
    gray = color.rgb2gray(image)
    gray_u8 = util.img_as_ubyte(gray)
    glcm = graycomatrix(
        gray_u8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        symmetric=True,
        normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    texture_features = np.hstack([graycoprops(glcm, p).flatten() for p in props])

    # 3. Shape descriptors
    binary = gray_u8 > 0
    regions = measure.regionprops(measure.label(binary))
    if not regions:
        shape_features = np.zeros(7, dtype=np.float32)
    else:
        r = max(regions, key=lambda reg: reg.area)
        area = r.area
        perimeter = r.perimeter
        ecc = r.eccentricity
        solidity = r.solidity
        extent = r.extent
        maj = r.major_axis_length
        min_ = r.minor_axis_length or 1.0
        aspect = maj / min_
        shape_features = np.array([area, perimeter, ecc, solidity, extent, maj, aspect], dtype=np.float32)

    # Concatenate all features
    features = np.concatenate([color_features, texture_features, shape_features], axis=0)

    return features

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

    # # Data Transform w/ feature extraction
    data_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    # # Data transform w/o feature extraction
    # data_transform = T.Compose([
    #     T.Resize((64, 64)),
    #     T.ToTensor(),
    # ])

    train_ds = PlantDiseaseDataset(train_df, transform=data_transform)
    test_ds  = PlantDiseaseDataset(test_df,  transform=data_transform)

    # feature_extraction_fn = extract_features_1
    # feature_extraction_fn.__name__ = "HOG_LBP" # Set the name for logging
    feature_extraction_fn = extract_features_2
    feature_extraction_fn.__name__ = "GLCM_Shape_Color" # Set the name for logging

    # Load and prepare data for SVM
    logger.info("Loading and preparing training data for SVM using feature extraction: %s", feature_extraction_fn.__name__)
    X_train, y_train = load_and_prepare_data(train_ds, feature_extraction_fn)
    logger.info(f"Training data shape: Features {X_train.shape}, Labels {y_train.shape}")

    logger.info("Loading and preparing test data for SVM...")
    X_test, y_test = load_and_prepare_data(test_ds, feature_extraction_fn)
    logger.info(f"Test data shape: Features {X_test.shape}, Labels {y_test.shape}")

    logger.info("Data preparation complete. Commencing SVM training...")

    # Scaler + SVM pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),               
        ('svc', SVC(class_weight='balanced', cache_size=5000, kernel='rbf'))
    ])

    # Hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=5,
        n_jobs=N_JOBS,
        verbose=3,
        return_train_score=True,
        pre_dispatch=f"{N_JOBS * 2}*n_jobs"   # throttle the number of launched jobs
    )
    logger.info("Starting grid search for hyperparameter tuning...")
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