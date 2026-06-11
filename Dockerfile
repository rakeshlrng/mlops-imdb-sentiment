# Use official Python slim image as base
FROM python:3.11-slim

# Set build-time argument for Hugging Face model
ARG HF_MODEL_NAME=rakeshlrng/imdb-sentiment

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ src/

# Set environment variables
ENV HF_MODEL_NAME=${HF_MODEL_NAME}
ENV INPUT_TEXT="This movie is fantastic"

# Default command
CMD ["python", "src/inference.py"]
