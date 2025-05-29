FROM python:3.10-slim-buster

WORKDIR /app

# Copy the requirements file into the container's working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire 'src' directory into /app/src
COPY src ./src

# Copy the entire 'outputs' directory into /app/outputs
COPY outputs ./outputs

EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]