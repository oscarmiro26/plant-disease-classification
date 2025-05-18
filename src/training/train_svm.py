import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
import torchvision.transforms as T

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config
from ..utils.logger import setup_logger


PARAM_GRID = {
    'svc__C': [0.001, 0.01, 0.1, 1, 10],
    'svc__gamma': ['scale', 'auto', 0.001, 0.0001],
}

# Logging setup
logger, RUN_DIR = setup_logger(
    model_name="svm",
    base_log_dir=config.LOG_DIR
)

def load_and_prepare_data(dataset: PlantDiseaseDataset):
    """Loading and preprocessing (normalization/standardization)."""
    all_images = []
    all_labels = []
    logger.info(f"Loading {len(dataset)} samples...")
    for img, label in tqdm(dataset, desc="Loading images"):
        img = img.numpy().astype(np.float32) / 255.0
        # Flatten into 1D feature vector
        all_images.append(img.reshape(-1))
        all_labels.append(label)
    X = np.stack(all_images, axis=0)
    y = np.array(all_labels)
    return X, y

def train_and_evaluate():
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

    # Resize for computational efficiency
    data_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
    ])
    train_ds = PlantDiseaseDataset(train_df, transform=data_transform)
    test_ds  = PlantDiseaseDataset(test_df,  transform=data_transform)

    # Load and prepare data for SVM
    logger.info("Loading and preparing training data for SVM...")
    X_train, y_train = load_and_prepare_data(train_ds)
    logger.info(f"Training data shape: Features {X_train.shape}, Labels {y_train.shape}")

    logger.info("Loading and preparing test data for SVM...")
    X_test, y_test = load_and_prepare_data(test_ds)
    logger.info(f"Test data shape: Features {X_test.shape}, Labels {y_test.shape}")

    logger.info("Data preparation complete. Commencing SVM training...")

    # Instantiate the SVM pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),               
        ('pca',    PCA(n_components=200)),                          
        ('svc',    SVC(class_weight='balanced', cache_size=5000, kernel='rbf'))
    ])

    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=5,
        n_jobs=-1,
        verbose=3,
        return_train_score=True,
    )

    # Fit the model using GridSearchCV using tqdm_joblib for progress bar
    n_candidates = len(list(ParameterGrid(PARAM_GRID)))
    total_fits   = n_candidates * grid_search.cv
    with tqdm_joblib(tqdm(desc="GridSearchCV", total=total_fits)):
        grid_search.fit(X_train, y_train)

    logger.info("Grid search complete.")
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    best_c = best_params['svc__C']
    best_gamma = best_params['svc__gamma']

    # Training
    logger.info("Training SVM model...")
    best_model.fit(X_train, y_train)
    logger.info("SVM training complete.")

    # Save the trained model
    model_name = f'svm_model_{best_c}_{best_gamma}'
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

    # Save metrics to CSV
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_test, y_pred,
        labels=labels_for_report,
        zero_division=0
    )
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