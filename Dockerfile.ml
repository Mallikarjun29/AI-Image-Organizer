FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt dvc

# Copy the ML pipeline code
COPY ml_pipeline /app/ml_pipeline
COPY drift_detect.py /app/
COPY hyperparameters.yaml /app/
COPY dvc.yaml /app/
COPY .dvc /app/.dvc

# Create necessary directories
RUN mkdir -p uploads models data/retraining/images

# Command to run drift detection
CMD ["python", "drift_detect.py"]