# Slim serving image: only the API code and its runtime deps (no training stack, no data).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime required by xgboost wheels on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached until requirements-serve.txt changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/__init__.py src/features.py src/model_utils.py src/api.py ./src/

# Never run as root inside the container
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/logs \
    && chown -R appuser:appuser /app
USER appuser

ENV MODEL_URI=models:/fraud-detector@champion \
    FRAUD_THRESHOLD=0.5 \
    PREDICTION_LOG_DIR=/app/data/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
