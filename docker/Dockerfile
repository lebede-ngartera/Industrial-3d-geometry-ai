FROM python:3.10-slim

WORKDIR /app

# System dependencies for PyTorch Geometric, Open3D, and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (using docker-specific requirements without PyG)
COPY docker/requirements-docker.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# Default: run demo
CMD ["python", "scripts/demo.py", "--device", "cpu"]
