# Stage 1: Build & Python env
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Metadata
LABEL maintainer="Sidharth Kumar Pradhan"
LABEL description="StagAI MAVIC-V2 - Competitive SAR Image Classification"

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies (basic utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Environment variables for CUDA/GPU
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Main executable. app.py will detect hardware and run locally or remotely.
CMD ["python", "app.py"]
