import os
import tempfile
import logging
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx

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
    version="1.0.0"
)

# --------------------------------------------------
# Security (HACKATHON COMPATIBLE)
# --------------------------------------------------
VALID_API_KEY = "hackathon_2024_secret_token"

def verify_token(x_api_key: str = Header(None)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Not authenticated")
    return x_api_key

# --------------------------------------------------
# Models
# --------------------------------------------------
class AudioRequest(BaseModel):
    audio_url: str = Field(..., description="Public audio URL")

class PredictionResponse(BaseModel):
    classification: str
    confidence: float
    language: str
    explanation: str

# --------------------------------------------------
# Audio download
# --------------------------------------------------
async def download_audio(url: str) -> Path:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download audio")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(r.content)
        tmp.close()
        return Path(tmp.name)

# --------------------------------------------------
# Feature extraction (SAFE – no librosa / no numba)
# --------------------------------------------------
def extract_features(path: Path) -> dict:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    energy = float(np.mean(audio ** 2))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))))

    peaks, _ = find_peaks(np.abs(audio), height=np.std(audio))
    peak_density = float(len(peaks) / len(audio))

    return {
        "energy": energy,
        "zcr": zcr,
        "peak_density": peak_density
    }

# --------------------------------------------------
# Analysis
# --------------------------------------------------
def analyze(features: dict) -> dict:
    score = 0.5
    reasons = []

    if features["peak_density"] < 0.01:
        score += 0.2
        reasons.append("uniform waveform structure")

    if features["zcr"] < 0.05:
        score += 0.2
        reasons.append("low zero-crossing rate")

    confidence = round(min(score, 1.0), 2)
    label = "AI Generated" if confidence > 0.55 else "Human"

    return {
        "classification": label,
        "confidence": confidence,
        "language": "English",
        "explanation": f"Analysis suggests {label.lower()} voice: {', '.join(reasons)}"
    }

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: AudioRequest, token: str = Depends(verify_token)):
    parsed = urlparse(req.audio_url)
    path = None
    try:
        if parsed.scheme in ("http", "https"):
            path = await download_audio(req.audio_url)
        else:
            raise HTTPException(status_code=400, detail="Invalid audio URL")

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
async def handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )
