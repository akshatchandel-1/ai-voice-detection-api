# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# 🔥 CRITICAL: Disable numba JIT globally (before app runs)
ENV NUMBA_DISABLE_JIT=1

# Install system dependencies required for librosa and audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create non-root user (optional but safe)
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose port (Render ignores EXPOSE but it's fine)
EXPOSE 8000

# Run the application (Render-compatible, dynamic port)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
