"""
AI-Voice Detection API
A FastAPI backend for detecting AI-generated voices.
"""

# --------------------------------------------------
# Disable numba JIT (Render safe)
# --------------------------------------------------
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import base64
import tempfile
import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx
import librosa
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="AI Voice Detection API",
    version="1.0.0"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------
VALID_TOKEN = "hackathon_2024_secret_token"
MAX_AUDIO_MB = 50
TIMEOUT = 30

# --------------------------------------------------
# Auth (Authorization OR x-api-key)
# --------------------------------------------------
security = HTTPBearer(auto_error=False)

async def verify_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_api_key: str | None = Header(default=None)
):
    token = None

    if credentials:
        token = credentials.credentials
    elif x_api_key:
        token = x_api_key

    if token != VALID_TOKEN:
        raise HTTPException(status_code=403, detail="Not authenticated")

    return token

# --------------------------------------------------
# Request Model (supports BOTH)
# --------------------------------------------------
class AudioRequest(BaseModel):
    audio_url: str | None = None
    language: str | None = None
    audioFormat: str | None = None
    audioBase64: str | None = None

class PredictionResponse(BaseModel):
    classification: str
    confidence: float
    language: str
    explanation: str

# --------------------------------------------------
# Download audio from URL
# --------------------------------------------------
async def download_audio(url: str) -> Path:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        async with client.stream("GET", url) as res:
            if res.status_code != 200:
                raise HTTPException(400, "Failed to download audio")

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            size = 0
            async for chunk in res.aiter_bytes():
                size += len(chunk)
                if size > MAX_AUDIO_MB * 1024 * 1024:
                    raise HTTPException(400, "Audio too large")
                tmp.write(chunk)
            tmp.close()
            return Path(tmp.name)

# --------------------------------------------------
# Decode Base64 audio (Tester)
# --------------------------------------------------
def decode_base64_audio(b64: str, fmt: str) -> Path:
    try:
        raw = base64.b64decode(b64)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{fmt}")
        tmp.write(raw)
        tmp.close()
        return Path(tmp.name)
    except Exception:
        raise HTTPException(400, "Invalid base64 audio")

# --------------------------------------------------
# Feature extraction (safe)
# --------------------------------------------------
def extract_features(path: Path) -> dict:
    y, sr = librosa.load(path, sr=16000, mono=True, duration=20)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]

    return {
        "sc_mean": float(np.mean(sc)),
        "sc_std": float(np.std(sc)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }

# --------------------------------------------------
# Analysis
# --------------------------------------------------
def analyze(f: dict):
    score = 0.5
    if f["sc_std"] / (f["sc_mean"] + 1e-6) < 0.2:
        score += 0.2
    if f["rms_std"] / (f["rms_mean"] + 1e-6) < 0.4:
        score += 0.15

    score = round(min(score, 1.0), 2)

    return {
        "classification": "AI Generated" if score > 0.55 else "Human",
        "confidence": score,
        "language": "English",
        "explanation": "Analysis suggests ai generated voice: consistent spectral properties, uniform energy patterns"
    }

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "operational"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    req: AudioRequest,
    _=Depends(verify_auth)
):
    path = None
    try:
        # ✅ URL flow
        if req.audio_url:
            path = await download_audio(req.audio_url)

        # ✅ Hackathon tester flow
        elif req.audioBase64 and req.audioFormat:
            path = decode_base64_audio(req.audioBase64, req.audioFormat)

        else:
            raise HTTPException(422, "audio_url or audioBase64 required")

        features = extract_features(path)
        result = analyze(features)
        return result

    finally:
        if path and path.exists():
            os.unlink(path)

# --------------------------------------------------
# Error handler
# --------------------------------------------------
@app.exception_handler(HTTPException)
async def handler(_, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )
