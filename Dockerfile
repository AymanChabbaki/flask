# Use lightweight Python image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for scikit-learn
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL files
COPY . .

# Run on port 8080 (Cloud Run requirement)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "main:app"]