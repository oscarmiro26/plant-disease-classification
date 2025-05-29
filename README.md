# Plant Disease Classification FastAPI Service

This service provides an API endpoint for classifying plant diseases from images using a selection of our ML models (incl. SVM, ResNet50, ResNet Ensemble).

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    # git clone <repository-url>
    # cd <repository-folder>
    ```

    NOTE: If the image folders in `/samples` are empty, try deleting and cloning the repo again, or just download them manually from the remote repo. 

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

## Running the Service

To start the FastAPI service, navigate to the root directory and run the `main.py` script using Uvicorn:

```bash
uvicorn src.main:app
```

The API will run on your local host: http://127.0.0.1:8000.

If it takes a while to load try refreshing. Once the API is up and running, go to '/docs' by typing it in the search bar.

## Using the API

Click on `POST` and `Try it out`. Then, enter the model type you want to use and upload an image (samples are available in `/samples'`. Download them first before you upload). Next, press `Execute` and you will see the prediction results under `Responses` along with the curl message.