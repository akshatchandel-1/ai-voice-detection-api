# ✅ Python 3.10 (numba + librosa compatible)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies for audio
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 🔥 Disable numba JIT BEFORE Python imports
ENV NUMBA_DISABLE_JIT=1

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Security: non-root user
RUN useradd -m apiuser
USER apiuser

# Expose port (Render ignores but OK)
EXPOSE 8000

# Render-compatible start
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
