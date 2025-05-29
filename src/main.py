import io
import os
import joblib
import uvicorn
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
from PIL import Image
import numpy as np
from .data.svm_preprocessing import segment_leaf, extract_features
from .training.config import INV_LABEL_MAP, MODELS_DIR
import torch
from torchvision import models
import torchvision.transforms as T
from .data.preprocessing import normalize_transform

# Define allowed image and model types globally
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/JPEG", "image/png"]
ALLOWED_MODELS = ["svm", "resnet"] ## Add other models here

class ModelInput(BaseModel):
    """
    Pydantic model for the input data to the prediction endpoint.
    Uses Annotated to specify that fields come from Form and File.
    Includes validators for model_type and file content_type.
    """
    model_type: Annotated[str, Form(description="Models to use.", example="svm")]
    file: Annotated[UploadFile, File(description="Image file to classify (JPEG or PNG)")]

    @validator('model_type')
    def validate_model_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Invalid model_type. Supported: {', '.join(ALLOWED_MODELS)}")
        return v

    @validator('file')
    def validate_file_type(cls, v: UploadFile) -> UploadFile:
        if not v.filename: # Check if a file was actually uploaded
             raise ValueError("No file uploaded.")
        if v.content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(
                f"Invalid file type: {v.content_type}. Supported types are: {', '.join(ALLOWED_IMAGE_TYPES)}"
            )
        return v

# Define the response model
class PredictionResponse(BaseModel):
    model: str = Field(..., example="svm or resnet", description="The type of model used for prediction.")
    prediction: str = Field(..., example="Apple___healthy", description="The predicted class or disease of the plant.")

# Global variable to hold models
svm_model = None 
resnet_model = None

# Create a FastAPI app instance
app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases from leaf images using a selection of our models (SVM, ResNet50).",
    author="Ravindra, Oscar, Benediktus, Richard",
    version="1.0.0",
)

@app.get(
    "/",
    tags=["General"],
    summary="Root Endpoint",
    description="Returns a welcome message indicating the API is running and provides a link to the API documentation."
)
async def root():
    """
    Root endpoint to check if the server is running.
    Returns a simple message indicating the server is up and provides a link to the docs.
    """
    return {"message": "Plant Disease Classification API is running. Visit /docs for API documentation."}

@app.on_event("startup")
async def load_model():
    """
    Load the models from the specified path when the application starts.
    """
    global svm_model
    global resnet_model
    # global ___ 
    try:
        svm_path = os.path.join(MODELS_DIR, "svm.pkl")
        svm_model = joblib.load(svm_path)
    except FileNotFoundError:
        print(f"Error: Model file {MODELS_DIR} not found.")
    except Exception as e:
        print(f"Error loading SVM model: {e}")
        # Handle other potential errors during model loading
    
    try:
        # Instantiate a ResNet50, adjust final layer for the number of classes
        base = models.resnet50(pretrained=False)
        num_classes = len(INV_LABEL_MAP)
        num_ftrs = base.fc.in_features
        base.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )
        # Load your trained weights
        resnet_model_path = os.path.join(MODELS_DIR, "resnet50_9897.pth")
        state = torch.load(resnet_model_path, map_location="cpu")
        base.load_state_dict(state)
        base.eval()
        resnet_model = base
    except FileNotFoundError:
        print(f"Error: Model file {resnet_model_path} not found.")
    except Exception as e:
        print(f"Error loading ResNet model: {e}")

@app.post(
    "/predict/",
    response_model=PredictionResponse,
    tags=["Classification"],
    summary="Predict Plant Disease",
    description="Upload an image of a plant leaf (JPEG or PNG) and specify the model type ('svm' / 'resnet') to classify its disease. The API will return the model used and the predicted disease class."
)
async def predict(input_data: ModelInput = Depends()):
    """
    Receives an image and a model type (via form data), predicts the class using the loaded model.
    Input validation (model_type, file type) is handled by the Pydantic ModelInput.
    """
    if svm_model is None and input_data.model_type == "svm":
        raise HTTPException(status_code=500, detail="SVM model not loaded. Check server logs.")
    if resnet_model is None and input_data.model_type == "resnet":
        raise HTTPException(status_code=500, detail="ResNet model not loaded. Check server logs.")

    # model_type and file are accessed via input_data.
    # Pydantic validators in ModelInput have already checked model_type and file.content_type.

    try:
        if input_data.model_type == "svm":
            # Load the SVM model from the joblib file
            model = svm_model
        elif input_data.model_type == "resnet":
            # Use the pre-loaded ResNet model
            model = resnet_model
        elif model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
        contents = await input_data.file.read()
        # Ensure file is not empty after reading
        if not contents:
            raise HTTPException(status_code=400, detail="File not loaded. Please upload a valid image file.")
        
        # Open the image using Pillow
        img = Image.open(io.BytesIO(contents))
        # Convert PIL image to NumPy array (RGB)
        img_np = np.array(img.convert("RGB"))
        
        # Segment the leaf if using SVM
        if input_data.model_type == "svm":
            segmented_img = segment_leaf(img_np)

    except HTTPException: # Re-raise HTTPExceptions explicitly
        raise
    except ValueError as ve: # Catch ValueErrors from Pydantic validators if not caught by FastAPI
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # If there's an error opening or processing the image, return an error response
        raise HTTPException(status_code=400, detail=f"Could not process image file: {e}")

    # Dispatch to the chosen model (Add predictions for your model here as well)
    try:
        if input_data.model_type == "svm":
            features = extract_features(segmented_img)
            pred = svm_model.predict([features])[0]
            class_label = INV_LABEL_MAP.get(pred, "Unknown class")

        if input_data.model_type == "resnet":  # resnet
            # Prepare pytorch input
            preprocess = T.Compose([
                T.ToPILImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize_transform,
            ])
            # segmented_img is a np.ndarray; convert and preprocess
            tensor = preprocess(img_np).unsqueeze(0)
            with torch.no_grad():
                outputs = resnet_model(tensor)
                _, idx = outputs.max(1)
            class_label = INV_LABEL_MAP.get(idx.item(), "Unknown class")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    # Return the model type and the prediction as a JSON response
    return JSONResponse(content={"model": input_data.model_type, "prediction": class_label})

if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn
    # reload=True enables auto-reloading when code changes (for development)
    uvicorn.run("src.main:app", reload=True)
