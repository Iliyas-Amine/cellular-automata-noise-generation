# 1. Base Image: Use a stable, lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libx11-6 \
    libglib2.0-0 \
    mesa-common-dev \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements.txt first (OPTIMIZATION: leverages Docker layer caching)
COPY requirements.txt .

# 5. Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the project files (scripts, logs folder structure, etc.)
# The '.' copies everything from the host directory into /app
COPY . .

# 7. Define the default command to run your primary script
CMD ["python", "main.py"]