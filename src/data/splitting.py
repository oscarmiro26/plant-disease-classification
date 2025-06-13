import os

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(data_dir, label_map, test_size, val_size, random_seed=None):
    """
    Scans the data directory, creates a DataFrame, and performs stratified data splits.
    """
    print(f"Scanning data directory: {data_dir}.")
    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        return None, None, None

    filepaths = []
    labels = []
    all_classes = sorted(label_map.keys())

    for class_name in all_classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"Class directory not found, skipping: {class_dir}.")
            continue

        try:
            image_files = [
                f
                for f in os.listdir(class_dir)
                if os.path.isfile(os.path.join(class_dir, f))
            ]
            for img_file in image_files:
                filepath = os.path.join(class_dir, img_file)
                filepaths.append(filepath)
                labels.append(class_name)
        except Exception as e:
            print(f"Error reading class directory {class_dir}: {e}.")
            continue

    if not filepaths:
        print("No files found in data directory.")
        return None, None, None

    full_df = pd.DataFrame({"filepath": filepaths, "label": labels})
    print(f"Found {len(full_df)} total images across {len(all_classes)} classes.")

    # Check if every class has enough data to perform splits (3 images minimum)
    min_samples = full_df["label"].value_counts().min()
    if min_samples < 3:
        print(f"Not enough samples for splits. Minimum samples: {min_samples}")

    print("Splitting data...")
    X = full_df["filepath"]
    y = full_df["label"]

    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        # Because now the proportion of data left is (1 - test_size), we need to adjust
        # the validation split size to match the original proportion
        remaining_proportion = 1 - test_size
        val_split_adjusted = val_size / remaining_proportion

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_split_adjusted,
            random_state=random_seed,
            stratify=y_train_val,
        )
    except Exception as e:
        print(f"Error during stratified splitting: {e}.")
        return None, None, None

    train_df = pd.DataFrame({"filepath": X_train, "label": y_train}).reset_index(
        drop=True
    )
    val_df = pd.DataFrame({"filepath": X_val, "label": y_val}).reset_index(drop=True)
    test_df = pd.DataFrame({"filepath": X_test, "label": y_test}).reset_index(drop=True)

    print("Splitting complete.")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")

    # Final check
    if train_df.empty or val_df.empty or test_df.empty:
        print("Splits are empty, something went wrong.")
        return None, None, None

    return train_df, val_df, test_df
