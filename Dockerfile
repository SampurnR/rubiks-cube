# Credit Card Fraud Detection - Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fraud_model.py .
COPY train_best_model.py .
COPY predict.py .

# Copy trained model (if exists)
COPY models/ models/

# Default command (predict mode)
CMD ["python", "predict.py", "--mode", "sample", "--samples", "10"]
