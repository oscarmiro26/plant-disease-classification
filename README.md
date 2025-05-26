# Plant Disease Classification FastAPI Service

This service provides an API endpoint for classifying plant diseases from images using a pre-trained scikit-learn SVM pipeline.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    # git clone <repository-url>
    # cd <repository-folder>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    Make sure you have `pip` installed. Then, from the project's root directory, run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the SVM model:**
    Ensure your trained SVM pipeline is saved as `svm.pkl` inside the `src/models/` directory. If the `src/models/` directory doesn't exist, create it.

## Running the Service

To start the FastAPI service, navigate to the root directory and run the `main.py` script using Uvicorn:

```bash
uvicorn src.main:app