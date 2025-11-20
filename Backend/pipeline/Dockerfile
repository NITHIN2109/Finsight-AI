FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# DATA_DIR default inside container (Cloud Run Job will mount a volume or push to GCS later)
ENV DATA_DIR=/data

CMD ["python", "main.py"]