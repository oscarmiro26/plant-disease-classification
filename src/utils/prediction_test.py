import joblib
import numpy as np
import os
from PIL import Image
from src.data.svm_preprocessing import extract_features_3
from src.training.config import INV_LABEL_MAP  # Corrected import path

# Define the path to the model
# Assuming this script is in src/models/ and svm.pkl is also in src/models/
MODEL_PATH = "src/models/svm.pkl"


def main():
    # --- 1. Load the trained SVM model ---
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded SVM model from: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print(
            "Please ensure 'svm.pkl' is in the same directory as this script or provide the correct path."
        )
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    # --- 2. Prepare sample input data ---
    image_path = "data/raw/segmented/Apple___Black_rot/0b8dabb7-5f1b-4fdc-b3fa-30b289707b90___JR_FrgE.S 3047_final_masked.jpg"
    img = Image.open(image_path)
    sample_features = extract_features_3(np.array(img.convert("RGB")))
    print(f"\nShape of sample features: {sample_features.shape}")

    # --- 3. Make predictions ---
    try:
        if sample_features.ndim == 1:
            # If it's a 1D array, reshape it to 2D with one sample
            sample_features = sample_features.reshape(1, -1)
        elif sample_features.ndim != 2 or sample_features.shape[0] != 1:
            print(
                "Warning: Sample features should be a 2D array with shape (1, D). Reshaping..."
            )
            sample_features = sample_features.reshape(1, -1)
        predictions = model.predict(sample_features)
    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")
        print("Ensure the dummy feature vector length matches what the model expects.")
        return

    # --- 4. Show the output of the predictions ---
    print("\n--- Prediction Output ---")
    if isinstance(predictions, np.ndarray):
        print(f"Type of predictions: {type(predictions)}")
        print(f"Shape of predictions array: {predictions.shape}")
        print("Predictions:")
        for i, p_int in enumerate(predictions):
            # Map integer prediction to string label
            p_label = INV_LABEL_MAP.get(p_int, f"Unknown label for int {p_int}")
            print(f"  Sample {i+1}: {p_int} -> '{p_label}' (Type: {type(p_int)})")
    else:
        print(f"Unexpected prediction output type: {type(predictions)}")
        print(f"Predictions: {predictions}")

    # Example of how it might be used for a single prediction (like in your FastAPI app)
    if sample_features.shape[0] > 0:
        single_feature_vector = sample_features[
            0:1
        ]  # Keep it as a 2D array with one row
        single_prediction_int = model.predict(single_feature_vector)[
            0
        ]  # Get the first element
        # Map integer prediction to string label
        single_prediction_label = INV_LABEL_MAP.get(
            single_prediction_int, f"Unknown label for int {single_prediction_int}"
        )
        print("\n--- Single Prediction Example (like in API) ---")
        print(f"Input features (first 5 elements): {single_feature_vector[0, :5]}...")
        print(
            f"Predicted class: {single_prediction_int} -> '{single_prediction_label}' (Type: {type(single_prediction_int)})"
        )


if __name__ == "__main__":
    main()
