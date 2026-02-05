"""
AI-Voice Detection API
Stable, hackathon-safe, Render-safe implementation
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from pathlib import Path
import httpx
import librosa
import numpy as np
import soundfile as sf
import tempfile
import os
import logging

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
security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if credentials.credentials != VALID_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authentication token")
    return credentials.credentials

# --------------------------------------------------
# Models
# --------------------------------------------------
class AudioRequest(BaseModel):
    audio_url: str = Field(..., description="Public audio URL (mp3/wav)")

class PredictionResponse(BaseModel):
    classification: str
    confidence: float
    language: str
    explanation: str

# --------------------------------------------------
# Audio download
# --------------------------------------------------
async def download_audio(url: str) -> Path:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(400, "Failed to download audio")

        suffix = ".wav" if ".wav" in url else ".mp3"
        f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        f.write(r.content)
        f.close()
        return Path(f.name)

# --------------------------------------------------
# 🔥 STABLE FEATURE EXTRACTION (NO NUMBA)
# --------------------------------------------------
def extract_audio_features(audio_path: Path) -> dict:
    try:
        y, sr = sf.read(audio_path, always_2d=False)

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        y = y[: sr * 20]  # max 20 seconds

        features = {}

        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        rms = librosa.feature.rms(y=y)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)

        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
        features["rms_mean"] = float(np.mean(rms))
        features["rms_std"] = float(np.std(rms))

        for i in range(5):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))

        return features

    except Exception as e:
        raise HTTPException(400, f"Failed to process audio file: {str(e)}")

# --------------------------------------------------
# Analysis
# --------------------------------------------------
def analyze_voice(features: dict):
    score = 0.5
    reasons = []

    if features["spectral_centroid_std"] < 0.15 * features["spectral_centroid_mean"]:
        score += 0.2
        reasons.append("consistent spectral properties")

    if features["rms_std"] < 0.4 * features["rms_mean"]:
        score += 0.15
        reasons.append("uniform energy patterns")

    confidence = round(min(max(score, 0), 1), 2)
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
async def predict_voice(
    request: AudioRequest,
    token: str = Depends(verify_token)
):
    audio_path = None
    try:
        audio_path = await download_audio(request.audio_url)
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
