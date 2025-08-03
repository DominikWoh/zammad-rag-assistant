# syntax=docker/dockerfile:1.7

# -------- Base builder to cache wheels (optional, minimal here) --------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System packages (minimal). Add more if some deps require build tools.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only backend requirements first for better caching
COPY web-ui/backend/requirements.txt ./web-ui/backend/requirements.txt

# Install pip deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ./web-ui/backend/requirements.txt

# -------- Final stage --------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV_FILE=/data/config/ticket_ingest.env \
    PROMPTS_DIR=/data/config/prompts \
    HUGGINGFACE_CACHE_DIR=/data/cache \
    EMBED_MODEL=intfloat/multilingual-e5-base \
    WEBUI_HOST=0.0.0.0 \
    WEBUI_PORT=5000

# Create runtime dirs
RUN mkdir -p /data/config /data/cache

# Install minimal system packages (add build-essential if needed by future deps)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed site-packages from base
COPY --from=base /usr/local /usr/local

# Copy the entire app
COPY . /app

# Preload SentenceTransformer model into cache during build to avoid downloads at runtime
# If the embedding model changes via build-arg/ENV, this step will re-run.
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.getenv('EMBED_MODEL','intfloat/multilingual-e5-base'), cache_folder=os.getenv('HUGGINGFACE_CACHE_DIR','/data/cache'))"

EXPOSE 5000

# Healthcheck (optional): simple TCP check against uvicorn
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c \"import socket,sys; s=socket.socket(); \
  s.settimeout(3); \
  s.connect(('127.0.0.1', int(os.environ.get('WEBUI_PORT','5000')))); \
  s.close(); \
  print('ok')\" || exit 1

# Start the FastAPI app using file-based import with app-dir to avoid module name issues
CMD ["python", "-m", "uvicorn", "backend.main:app", "--app-dir", "web-ui", "--host", "0.0.0.0", "--port", "5000"]