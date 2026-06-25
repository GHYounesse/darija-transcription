FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create directory for models
RUN mkdir -p /app/models

# Expose port 7860 (HF Spaces default)
EXPOSE 7860

# Run FastAPI on port 7860
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]