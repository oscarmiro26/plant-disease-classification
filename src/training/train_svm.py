import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog, local_binary_pattern
import joblib
import torchvision.transforms as transforms

from ..data.splitting import create_splits
from ..data.datasets import PlantDiseaseDataset
from ..training import config
from ..utils.logger import setup_logger

# SVM Hyperparameters (Grid Search)
PARAM_GRID = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.001, 0.0001],
    'kernel': ['rbf']
}

# Logging setup
logger, RUN_DIR = setup_logger(
    model_name="svm",
    base_log_dir=config.LOG_DIR
)

# Transforms for image data - consistent for SVM feature extraction
data_transform = transforms.Compose([
    transforms.ToTensor(),
])

def extract_features(image):
    """
    Extracts Histogram of Oriented Gradients (HOG) and Local Binary Pattern (LBP) features from an image. Concatenates them into a single feature vector.
    """
    hog_features = hog(image, orientations=8, pixels_per_cell=(16,16),
                      cells_per_block=(1,1), visualize=False)
    # Example LBP features
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, 10))
    return np.concatenate([hog_features, hist])

def load_and_prepare_data(dataset: PlantDiseaseDataset):
    """Loads all data from a PlantDiseaseDataset and prepares it for SVM."""
    all_features = []
    all_labels = []
    logger.info(f"Loading {len(dataset)} samples...")
    for i in range(len(dataset)):
        img_tensor, label = dataset[i]

        # Feature extraction
        image_np = img_tensor.squeeze(0).numpy() # Convert to 2D np array
        features = extract_features(image_np)

        # Flatten the image tensor to create a feature vector
        all_features.append(features)
        all_labels.append(label)
        if (i + 1) % 500 == 0: # Log progress
            logger.info(f"Loaded {i+1}/{len(dataset)} samples.")
    
    return np.array(all_features), np.array(all_labels)

def train_and_evaluate():
    logger.info(f"Starting SVM training and evaluation run in {RUN_DIR}")

    # Data splits
    # We'll use train_df for training and test_df for final evaluation.
    # val_df could be used for hyperparameter tuning if implemented separately.
    logger.info("Creating data splits...")
    train_df, val_df, test_df = create_splits(
        data_dir=config.RAW_DATA_DIR,
        label_map=config.LABEL_MAP,
        test_size=config.TEST_SPLIT_SIZE,
        val_size=config.VALIDATION_SPLIT_SIZE,
    )

    # Combine train and validation sets for SVM training
    train_df = pd.concat([train_df, val_df], ignore_index=True)

    # Datasets
    logger.info("Creating datasets...")
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
    # Model: Support Vector Machine Classifier
    grid_search = GridSearchCV(
        svm.SVC(class_weight='balanced', random_state=42),
        PARAM_GRID,
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,
        verbose=3,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_c = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    
    logger.info("Grid search complete.")
    logger.info(f"Best parameters found: C={best_c}, gamma={best_gamma}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
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


