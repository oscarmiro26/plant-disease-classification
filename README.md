# Plant Disease Classification FastAPI Service

This service provides an API endpoint for classifying plant diseases from images using a selection of our ML models (incl. SVM and ResNet50).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

    NOTE: If the image folders in `/samples` are empty, try deleting and cloning the repo again, or just download them manually from the remote repo.

2.  **Ensure model files are present:**
    Make sure your trained model files (`svm.pkl`, `resnet50_9897.pth`, etc.) are located in the `outputs/models/` directory within your project structure. This is essential for the service to load them correctly. Your project structure should look like this:

    ```
    plant-disease-classification/ <-- This is your project root
    |-- requirements.txt
    |-- outputs/
    |   |-- models/
    |       |-- svm.pkl
    |       |-- resnet50_9897.pth
    |-- src/
        |-- ... (other directories like data, training)
        |-- main.py
    ```

## Running the Service

You can run the service directly using a virtual environment or by using Docker.

### Option 1: Using a Virtual Environment (Local Development)

1.  **Create a virtual environment:**
    From your project's root directory (`main/`), run:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    (Note: many of the project dependencies are not compatible with Python versions above 3.11 or below 3.7.)

2.  **Install dependencies:**
    Make sure you have `pip` installed. From the project's root directory (`main/`), run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the FastAPI service:**
    From the project's root directory (`main/`), run:
    ```bash
    uvicorn src.main:app --host 0.0.0.0 --port 8000
    ```
    (Note: `reload=True` is typically for development and isn't included here for production-like instructions)

### Option 2: Using Docker (Recommended for Deployment/Consistency)

Docker provides a consistent environment for running the application without worrying about local dependencies.

1.  **Ensure Docker is installed:**
    If you don't have Docker, download and install it from [https://www.docker.com/get-started](https://www.docker.com/get-started).

2.  **Build the Docker Image:**
    Open your terminal, navigate to the **root directory of this project (`main/`)** (where the `Dockerfile` is located), and run:
    ```bash
    docker build -t plant-disease-classifier .
    ```
    This command builds a Docker image named `plant-disease-classifier` based on the `Dockerfile` in the current directory.

3.  **Run the Docker Container:**
    Once the image is built, you can run a container from it:
    ```bash
    docker run -p 8000:8000 plant-disease-classifier
    ```
    * `-p 8000:8000`: This maps port 8000 on your host machine to port 8000 inside the Docker container, allowing you to access the API.

The API will run on your local host: http://127.0.0.1:8000.

If it takes a while to load try refreshing. Once the API is up and running, go to '/docs' by typing it in the search bar.

## Using the API

Click on `POST` and `Try it out`. Then, enter the model type you want to use and upload an image (samples are available in `/samples'`. Download them first before you upload). Next, press `Execute` and you will see the prediction results under `Responses` along with the curl message.