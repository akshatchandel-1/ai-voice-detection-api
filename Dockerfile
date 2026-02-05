FROM python:3.10-slim

WORKDIR /app

# System deps for librosa
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Security
RUN useradd -m apiuser
USER apiuser

# Prevent numba JIT crash
ENV NUMBA_DISABLE_JIT=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
