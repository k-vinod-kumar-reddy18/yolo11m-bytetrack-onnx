FROM python:3.11-slim

WORKDIR /app

# OpenCV and video dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create result directories
RUN mkdir -p \
    05_results/pytorch \
    05_results/onnx \
    05_results/reports \
    06_logs

# Default command
CMD ["python", "03_bytetrack/run_bytetrack.py"]