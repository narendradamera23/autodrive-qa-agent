FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy all project files
COPY . .

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Default command (FastAPI)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]