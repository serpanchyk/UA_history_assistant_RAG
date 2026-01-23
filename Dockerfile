# Use a slim version of Python for a smaller image size
FROM python:3.11-slim

# Install system dependencies required for OpenCV, PyTorch, and pyzbar
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your source code
COPY . .

# Set the Python path so imports like 'src.logger' work correctly
ENV PYTHONPATH=/app

# Start the ingestion and indexing process
CMD ["python", "main.py"]