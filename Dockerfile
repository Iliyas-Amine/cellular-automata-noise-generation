FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for VTK and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libx11-6 \
    libglib2.0-0 \
    mesa-common-dev \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

CMD ["python", "main.py"]