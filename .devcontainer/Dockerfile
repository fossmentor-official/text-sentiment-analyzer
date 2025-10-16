# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    wget \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch CPU-only
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Optional dev tools
RUN pip install streamlit fastapi uvicorn

# Keep container running
CMD ["sleep", "infinity"]
