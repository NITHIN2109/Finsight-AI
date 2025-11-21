# Backend/Dockerfile

FROM python:3.13-slim

# Create app folder
WORKDIR /app

# Install system dependencies (for pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1) Install Python deps from server/requirements.txt
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy server code into /app/server
COPY server /app/server

# 3) Copy shared data folder into /app/data
COPY data /app/data

# Set workdir to server
WORKDIR /app/server

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
