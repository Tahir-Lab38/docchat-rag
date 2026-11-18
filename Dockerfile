# ============================================
# DocChat — Dockerfile
# ============================================
FROM python:3.11-slim

LABEL maintainer="DocChat"
LABEL description="AI-powered document Q&A assistant with RAG, citations, and hallucination prevention"
LABEL version="1.0.0"

# Keep Python quiet and store the embedding model cache outside the app tree.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    TOKENIZERS_PARALLELISM=false

# curl is used by the health check; libgomp1 is required by PyTorch.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only PyTorch wheel first so the image does not pull in
# several gigabytes of unused CUDA libraries.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so container startup is fast and offline.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# Copy application code.
COPY app/ ./app/
COPY frontend/ ./frontend/

# Create runtime data directories.
RUN mkdir -p /app/data /app/uploads

EXPOSE 8000

# Health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
