FROM python:3.11-slim AS base
WORKDIR /app

# Install system deps (for torch / numpy / scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install the trendzap_intelligence package from src/
RUN pip install --no-cache-dir -e .

EXPOSE 8000
# Railway injects $PORT — fall back to 8000 for local dev
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
