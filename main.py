"""
AI-Voice Detection API
A FastAPI backend for detecting AI-generated voices from audio URLs.
"""

import os
import tempfile
import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import soundfile as sf

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App init
# --------------------------------------------------
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects whether audio is AI-generated or human voice",
    version="1.0.0"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
VALID_BEARER_TOKEN = "hackathon_2024_secret_token"
MAX_AUDIO_SIZE_MB = 50
TIMEOUT_SECONDS = 30

# --------------------------------------------------
# Security
# --------------------------------------------------
from fastapi import Header

async def verify_token(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None)
):
    token = None

    # Case 1: Normal Authorization header
    if authorization and authorization.startswith("Bearer "):
        token = authorization.replace("Bearer ", "").strip()

    # Case 2: Hackathon tester (x-api-key)
    elif x_api_key:
        token = x_api_key.strip()

    if token != VALID_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Not authenticated")

    return token


# --------------------------------------------------
# Models
# --------------------------------------------------
class AudioRequest(BaseModel):
    audio_url: str = Field(..., description="Public HTTP/HTTPS URL to audio file")

class PredictionResponse(BaseModel):
    classification: str
    confidence: float
    language: str
    explanation: str

# --------------------------------------------------
# Download audio
# --------------------------------------------------
async def download_audio(url: str) -> Path:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        async with client.stream("GET", url) as r:
            if r.status_code != 200:
                raise HTTPException(400, "Failed to download audio")

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            size = 0
            async for chunk in r.aiter_bytes():
                size += len(chunk)
                if size > MAX_AUDIO_SIZE_MB * 1024 * 1024:
                    raise HTTPException(400, "Audio too large")
                tmp.write(chunk)

            tmp.close()
            return Path(tmp.name)

# --------------------------------------------------
# Feature extraction (NO librosa, NO numba)
# --------------------------------------------------
def extract_audio_features(path: Path) -> dict:
    try:
        y, sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)

        y = y[: sr * 20]  # max 20 sec

        rms = np.sqrt(np.mean(y ** 2))
        zcr = np.mean(np.abs(np.diff(np.sign(y)))) / 2

        spectrum = np.abs(np.fft.rfft(y))
        centroid = np.sum(spectrum * np.arange(len(spectrum))) / np.sum(spectrum)

        return {
            "rms": float(rms),
            "zcr": float(zcr),
            "centroid": float(centroid)
        }

    except Exception as e:
        raise HTTPException(400, f"Failed to process audio file: {e}")

# --------------------------------------------------
# Analysis
# --------------------------------------------------
def analyze_voice(features: dict):
    score = 0.5
    reasons = []

    if features["rms"] < 0.05:
        score += 0.2
        reasons.append("uniform energy patterns")

    if features["zcr"] < 0.08:
        score += 0.15
        reasons.append("stable waveform")

    confidence = round(min(score, 1.0), 2)
    classification = "AI Generated" if confidence > 0.55 else "Human"

    return classification, confidence, reasons

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
async def root():
    return {"status": "operational", "service": "AI Voice Detection API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    req: AudioRequest,
    token: str = Depends(verify_token)
):
    audio_path = None
    parsed = urlparse(req.audio_url)

    try:
        audio_path = await download_audio(req.audio_url)
        features = extract_audio_features(audio_path)
        cls, conf, reasons = analyze_voice(features)

        return {
            "classification": cls,
            "confidence": conf,
            "language": "English",
            "explanation": f"Analysis suggests {cls.lower()} voice: {', '.join(reasons)}"
        }

    finally:
        if audio_path and audio_path.exists():
            os.unlink(audio_path)

# --------------------------------------------------
# Error handler
# --------------------------------------------------
@app.exception_handler(HTTPException)
async def handler(_, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )
