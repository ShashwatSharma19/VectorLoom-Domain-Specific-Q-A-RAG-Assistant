# ── Stage 1: Base image with Python ─────────────────────────────
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# ── Stage 2: Install dependencies ───────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Copy application code ──────────────────────────────
COPY . .

# HuggingFace model cache: models will be stored in a Docker volume
# so they persist across container restarts (not re-downloaded each time)
ENV HF_HOME=/app/.cache/huggingface

# ── Default command (overridden by docker-compose) ───────────────
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
