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
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
import torchvision.transforms as T

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..data.svm_preprocessing import (
    extract_features,
    load_and_prepare_data,
)
from ..training import config
from ..utils.logger import setup_logger


N_JOBS = int(os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count()))
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

PARAM_GRID = {
    'svc__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'svc__gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
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

    # Define data transformations
    data_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    # Create datasets
    train_ds = PlantDiseaseDataset(train_df, transform=data_transform)
    test_ds  = PlantDiseaseDataset(test_df,  transform=data_transform)
    
    # Feature extraction function
    feature_extraction_fn = extract_features
    feature_extraction_fn.__name__ = "Gabor_HSV" # Set the name for logging

    # Load and prepare data for SVM
    logger.info("Loading and preparing training data for SVM using feature extraction: %s", feature_extraction_fn.__name__)
    X_train, y_train = load_and_prepare_data(train_ds, feature_extraction_fn, n_jobs=N_JOBS)
    logger.info(f"Training data shape: Features {X_train.shape}, Labels {y_train.shape}")
    logger.info("Loading and preparing test data for SVM...")
    X_test, y_test = load_and_prepare_data(test_ds, feature_extraction_fn, n_jobs=N_JOBS)
    logger.info(f"Test data shape: Features {X_test.shape}, Labels {y_test.shape}")

    logger.info("Data preparation complete. Commencing SVM training...")

    # Define the SVM pipeline
    # Using StandardScaler, VarianceThreshold, CorrelationFilter, PCA, and SVC
    logger.info("Setting up SVM pipeline with preprocessing and model...")
    pipeline = Pipeline([
        ('scaler',          StandardScaler()),
        ('var_thresh',      VarianceThreshold(threshold=1e-3)),
        ('corr_filter',     CorrelationFilter(threshold=0.9)),
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


if __name__ == "__main__":
    train_and_evaluate()