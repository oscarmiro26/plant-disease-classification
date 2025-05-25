import io
import joblib
import uvicorn
from typing import Annotated # Added import

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends # Added Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator # Added validator
from PIL import Image
import numpy as np
from .data.svm_preprocessing import extract_features_3

# Define allowed image types globally
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/JPEG", "image/png"]

class ModelInput(BaseModel):
    """
    Pydantic model for the input data to the prediction endpoint.
    Uses Annotated to specify that fields come from Form and File.
    Includes validators for model_type and file content_type.
    """
    model_type: Annotated[str, Form()]
    file: Annotated[UploadFile, File()]

    @validator('model_type')
    def validate_model_type(cls, v: str) -> str:
        if v.lower() != "svm":
            raise ValueError("Invalid model_type. Only 'svm' is supported.")
        return v.lower()

    @validator('file')
    def validate_file_type(cls, v: UploadFile) -> UploadFile:
        if not v.filename: # Check if a file was actually uploaded
             raise ValueError("No file uploaded.")
        if v.content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(
                f"Invalid file type: {v.content_type}. Supported types are: {', '.join(ALLOWED_IMAGE_TYPES)}"
            )
        return v

# Global variable to hold the SVM model
svm_model = None # Corrected from svm_pipeline to svm_model for consistency

# Create a FastAPI app instance
app = FastAPI()

@app.get("/")
async def root():
    """
    Root endpoint to check if the server is running.
    Returns a simple message indicating the server is up.
    """
    return {"message": "Plant Disease Classification API is running."}

@app.on_event("startup")
async def load_model():
    """
    Load the SVM pipeline from the specified path when the application starts.
    """
    global svm_model
    try:
        # Load the pre-trained SVM pipeline using joblib
        svm_model = joblib.load("src/models/svm.pkl")
    except FileNotFoundError:
        print("Error: Model file 'src/models/svm.pkl' not found.")
    except Exception as e:
        print(f"Error loading SVM model: {e}")
        # Handle other potential errors during model loading

@app.post("/predict/")
async def predict(input_data: ModelInput = Depends()):
    """
    Receives an image and a model type (via form data), predicts the class using the loaded SVM pipeline.
    Input validation (model_type, file type) is handled by the Pydantic ModelInput.
    """
    if svm_model is None:
        raise HTTPException(status_code=503, detail="SVM model not loaded. Check server logs.")

    # model_type and file are accessed via input_data.
    # Pydantic validators in ModelInput have already checked model_type and file.content_type.

    try:
        # Read the image file bytes
        contents = await input_data.file.read()
        # Ensure file is not empty after reading
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
        # Open the image using Pillow
        img = Image.open(io.BytesIO(contents))
        # Convert PIL image to NumPy array (RGB)
        img_np = np.array(img.convert("RGB"))

    except HTTPException: # Re-raise HTTPExceptions explicitly
        raise
    except ValueError as ve: # Catch ValueErrors from Pydantic validators if not caught by FastAPI
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # If there's an error opening or processing the image, return an error response
        raise HTTPException(status_code=400, detail=f"Could not process image file: {e}")

    try:
        # Extract features using the same function used during training
        features = extract_features_3(img_np)
        # Make a prediction using the SVM pipeline
        prediction = svm_model.predict([features])[0]
    except Exception as e:
        # Handle errors during the prediction phase
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    # Return the model type and the prediction as a JSON response
    return JSONResponse(content={"model": input_data.model_type, "prediction": prediction})

if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn
    # host="0.0.0.0" makes the server accessible externally
    # reload=True enables auto-reloading when code changes (for development)
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
