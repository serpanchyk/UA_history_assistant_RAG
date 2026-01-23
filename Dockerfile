# Use a runtime image (smaller than 'devel') compatible with CUDA 12.x
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python and essential system libraries for OpenCV/ZBar
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN ln -s /usr/bin/python3.11 /usr/bin/python

COPY requirements.txt .

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
CMD ["python", "main.py"]