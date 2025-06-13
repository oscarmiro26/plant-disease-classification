import io
import os
import joblib
import uvicorn
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator
from PIL import Image
import numpy as np
from .data.svm_preprocessing import segment_leaf, extract_features
from .training.config import INV_LABEL_MAP, MODELS_DIR
import torch
from torchvision import models
import torchvision.transforms as T
from .data.preprocessing import normalize_transform

ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/JPEG", "image/png"]
ALLOWED_MODELS = ["svm", "resnet"]  # Add other models here


class ModelInput(BaseModel):
    model_type: Annotated[str, Form(...)]
    file: Annotated[UploadFile, File(...)]

    @field_validator("model_type", mode="before")
    @classmethod
    def check_model_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ALLOWED_MODELS:
            raise ValueError(
                f"Invalid model_type. Supported: {', '.join(ALLOWED_MODELS)}"
            )
        return v

    @field_validator("file", mode="before")
    @classmethod
    def check_file_type(cls, v):
        if not v.filename:
            raise ValueError("No file uploaded.")
        if v.content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(
                f"Invalid file type: {v.content_type}. Supported: {', '.join(ALLOWED_IMAGE_TYPES)}"
            )
        return v


class PredictionResponse(BaseModel):
    model: str = Field(
        ...,
        example="svm or resnet",
        description="The type of model used for prediction.",
    )
    prediction: str = Field(
        ...,
        example="Apple___healthy",
        description="The predicted class or disease of the plant.",
    )


svm_model = None
resnet_model = None

app = FastAPI(
    title="Plant Disease Classification API",
    description="API for classifying plant diseases from leaf images using a selection of our models (SVM, ResNet50).",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.get("/", tags=["General"], summary="Get API Status")
async def root():
    return {
        "message": "Plant Disease Classification API is running. Visit /docs for API documentation."
    }


@app.on_event("startup")
async def load_model():
    global svm_model, resnet_model

    try:
        svm_path = os.path.join(MODELS_DIR, "svm.pkl")
        svm_model = joblib.load(svm_path)
    except FileNotFoundError:
        print(f"Error: Model file {svm_path} not found.")
    except Exception as e:
        print(f"Error loading SVM model: {e}")

    try:
        base = models.resnet50(pretrained=False)
        num_classes = len(INV_LABEL_MAP)
        num_ftrs = base.fc.in_features
        base.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )
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
)
async def predict(input_data: Annotated[ModelInput, Depends()]):
    if svm_model is None and input_data.model_type == "svm":
        raise HTTPException(
            status_code=500, detail="SVM model not loaded. Check server logs."
        )
    if resnet_model is None and input_data.model_type == "resnet":
        raise HTTPException(
            status_code=500, detail="ResNet model not loaded. Check server logs."
        )

    try:
        contents = await input_data.file.read()
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="File not loaded. Please upload a valid image file.",
            )
        img = Image.open(io.BytesIO(contents))
        img_np = np.array(img.convert("RGB"))

        if input_data.model_type == "svm":
            segmented_img = segment_leaf(img_np)
            features = extract_features(segmented_img)
            pred = svm_model.predict([features])[0]
            class_label = INV_LABEL_MAP.get(pred, "Unknown class")

        elif input_data.model_type == "resnet":
            preprocess = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
            tensor = preprocess(img_np).unsqueeze(0)
            with torch.no_grad():
                outputs = resnet_model(tensor)
                _, idx = outputs.max(1)
            class_label = INV_LABEL_MAP.get(idx.item(), "Unknown class")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    response = JSONResponse(
        content={"model": input_data.model_type, "prediction": class_label},
        status_code=200,
    )
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


if __name__ == "__main__":
    uvicorn.run("src.main:app", reload=True)
import io
import os
import joblib
import uvicorn
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import (
    BaseModel,
    field_validator,
    Field,
    ValidationError,
)  # Removed 'validator' as it's not used
from PIL import Image
import numpy as np

# Assuming these are correctly set up as local packages
# Ensure src/data/__init__.py and src/training/__init__.py exist
from .data.svm_preprocessing import segment_leaf, extract_features
from .training.config import INV_LABEL_MAP, MODELS_DIR
import torch
from torchvision import models
import torchvision.transforms as T
from .data.preprocessing import normalize_transform

# Define allowed image and model types globally
ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/JPEG", "image/png"]
ALLOWED_MODELS = ["svm", "resnet"]  ## Add other models here


class ModelInput(BaseModel):
    """
    Pydantic model for the input data to the prediction endpoint.
    Uses Annotated to specify that fields come from Form and File.
    Includes validators for model_type and file content_type.
    """

    model_type: Annotated[
        str,
        Form(
            ...,
            title="Model Selection",
            description="Choose which model to use for inference",
            example="resnet",
        ),
    ]
    file: Annotated[
        UploadFile,
        File(
            ...,
            title="Leaf Image",
            description="Upload a JPEG or PNG of the plant leaf",
            example="example_leaf.jpg",
        ),
    ]

    @field_validator("model_type", mode="before")
    @classmethod
    def check_model_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ALLOWED_MODELS:
            raise ValueError(
                f"Invalid model_type. Supported: {', '.join(ALLOWED_MODELS)}"
            )
        return v

    @field_validator("file", mode="before")
    @classmethod
    def check_file_type(cls, v: UploadFile):
        if not v.filename:
            raise ValueError("No file uploaded.")
        # Ensure that content_type is checked only if it's not None
        if v.content_type is None or v.content_type not in ALLOWED_IMAGE_TYPES:
            raise ValueError(
                f"Invalid file type: {v.content_type}. Supported: {', '.join(ALLOWED_IMAGE_TYPES)}"
            )
        return v


# Define the response model
class PredictionResponse(BaseModel):
    model: str = Field(
        ...,
        example="svm or resnet",
        description="The type of model used for prediction.",
    )
    prediction: str = Field(
        ...,
        example="Apple___healthy",
        description="The predicted class or disease of the plant.",
    )


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """
    Handles validation errors that occur during request processing (e.g., Pydantic validation errors
    for request body/form data).
    """
    return JSONResponse(
        status_code=422,  # 422 Unprocessable Entity is standard for validation errors
        content={
            "detail": [
                {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
                for err in exc.errors()
            ]
        },
    )


@app.exception_handler(ValidationError)
async def pydantic_error_handler(request, exc):
    """
    Handles Pydantic's ValidationError, which might occur in contexts not directly covered
    by RequestValidationError (e.g., if you manually validate a Pydantic model elsewhere).
    """
    return JSONResponse(
        status_code=400,
        content={
            "detail": [
                {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
                for err in exc.errors()
            ]
        },
    )


@app.get(
    "/",
    tags=["General"],
    summary="Get API Status",
    description=(
        "Returns 200 if the service is live.  Use this endpoint to check "
        "that the API is running and to discover your documentation URL."
    ),
    responses={
        200: {
            "description": "API is up and running",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Plant Disease Classification API is running. Visit /docs for API documentation."
                    }
                }
            },
        }
    },
)
async def root():
    return {
        "message": "Plant Disease Classification API is running. Visit /docs for API documentation."
    }


@app.on_event("startup")
async def load_model():
    """
    Load the models from the specified path when the application starts.
    """
    global svm_model
    global resnet_model
    try:
        svm_path = os.path.join(MODELS_DIR, "svm.pkl")
        if os.path.exists(svm_path):
            svm_model = joblib.load(svm_path)
            print(f"SVM model loaded successfully from {svm_path}")
        else:
            print(
                f"Warning: SVM model file not found at {svm_path}. SVM predictions will not be available."
            )
            svm_model = None  # Ensure it's explicitly None if not found
    except Exception as e:
        print(f"Error loading SVM model from {svm_path}: {e}")
        svm_model = None  # Ensure it's explicitly None on error

    try:
        # Instantiate a ResNet50, adjust final layer for the number of classes
        base = models.resnet50(weights=None)  # Set weights to None initially
        num_classes = len(INV_LABEL_MAP)
        num_ftrs = base.fc.in_features
        base.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes),
        )
        # Load your trained weights
        resnet_model_path = os.path.join(MODELS_DIR, "resnet50_9897.pth")
        if os.path.exists(resnet_model_path):
            state = torch.load(resnet_model_path, map_location="cpu")
            base.load_state_dict(state)
            base.eval()
            resnet_model = base
            print(f"ResNet model loaded successfully from {resnet_model_path}")
        else:
            print(
                f"Warning: ResNet model file not found at {resnet_model_path}. ResNet predictions will not be available."
            )
            resnet_model = None  # Ensure it's explicitly None if not found
    except Exception as e:
        print(f"Error loading ResNet model from {resnet_model_path}: {e}")
        resnet_model = None  # Ensure it's explicitly None on error


@app.post(
    "/predict/",
    response_model=PredictionResponse,
    tags=["Classification"],
    summary="Predict Plant Disease",
    description=(
        "This tool uses Machine Learning models to classify plant diseases from leaf images.\n\n"
        "It provides two different models: SVM and ResNet50.\n\n"
        "To choose a model, use the `model_type` form field, and input `svm` or `resnet`.\n\n"
        "Then it allows you to upload an image.\n\n"
        "To upload an image, use the `file` form field, and input a JPEG or PNG image file.\n\n"
        "The API will return a prediction with the label, which consists of the plant AND the disease.\n\n"
        "The label is in the format of 'Plant___Disease', to allow for easy parsing."
    ),
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {"model": "resnet", "prediction": "Tomato___Late_blight"}
                }
            },
        },
        400: {
            "description": "Bad request—invalid or missing input",
            "content": {
                "application/json": {
                    "example": {"detail": "Could not process image file: ..."}
                }
            },
        },
        422: {
            "description": "Validation Error (invalid form data)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "model_type"],
                                "msg": "field required",
                                "type": "value_error.missing",
                            }
                        ]
                    }
                }
            },
        },
        500: {"description": "Server error, model not loaded or processing failed"},
    },
)
async def predict(input_data: ModelInput = Depends()):
    """
    Receives an image and a model type (via form data), predicts the class using the loaded model.
    Input validation (model_type, file type) is handled by the Pydantic ModelInput.
    """
    if input_data.model_type == "svm" and svm_model is None:
        raise HTTPException(
            status_code=500, detail="SVM model not loaded. Check server startup logs."
        )
    if input_data.model_type == "resnet" and resnet_model is None:
        raise HTTPException(
            status_code=500,
            detail="ResNet model not loaded. Check server startup logs.",
        )

    # model_type and file are accessed via input_data.
    # Pydantic validators in ModelInput have already checked model_type and file.content_type.

    model = None
    if input_data.model_type == "svm":
        model = svm_model
    elif input_data.model_type == "resnet":
        model = resnet_model

    if model is None:
        # This case should ideally be caught by the earlier checks for model availability
        # but provides an extra layer of safety.
        raise HTTPException(
            status_code=500,
            detail=f"Selected model '{input_data.model_type}' is not loaded.",
        )

    try:
        contents = await input_data.file.read()
        # Ensure file is not empty after reading
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty. Please upload a valid image file.",
            )

        # Open the image using Pillow
        img = Image.open(io.BytesIO(contents))
    except HTTPException:
        # Re-raise HTTPExceptions (e.g., empty file)
        raise
    except Exception as e:
        # Catch any other errors during file reading/image opening
        raise HTTPException(
            status_code=400, detail=f"Could not process image file: {e}"
        )

    # Convert PIL image to NumPy array (RGB)
    img_np = np.array(img.convert("RGB"))

    # Dispatch to the chosen model
    class_label = "Unknown"  # Default value

    try:
        if input_data.model_type == "svm":
            # Segment the leaf if using SVM
            segmented_img = segment_leaf(img_np)
            features = extract_features(segmented_img)
            # Ensure features is a 2D array for prediction if it's a single sample
            if features.ndim == 1:
                features = features.reshape(1, -1)
            pred = svm_model.predict(features)[0]
            class_label = INV_LABEL_MAP.get(pred, "Unknown class (SVM)")

        elif input_data.model_type == "resnet":
            # Prepare pytorch input
            preprocess = T.Compose(
                [
                    T.ToPILImage(),
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
            # img_np is a np.ndarray; convert and preprocess
            tensor = preprocess(img_np).unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                outputs = resnet_model(tensor)
                _, idx = outputs.max(1)
            class_label = INV_LABEL_MAP.get(idx.item(), "Unknown class (ResNet)")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction for {input_data.model_type} model: {e}",
        )

    response = JSONResponse(
        content={"model": input_data.model_type, "prediction": class_label},
        status_code=200,
    )
    response.headers["Cache-Control"] = "public, max-age=3600"
    return response


if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn
    # reload=True enables auto-reloading when code changes (for development)
    # Ensure this script is run as 'python -m src.main' or directly from the src directory
    # For direct execution, '__main__:app' refers to the 'app' object in the current script.
    uvicorn.run("__main__:app", host="0.0.0.0", port=8000, reload=True)
