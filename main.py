"""
AI-Voice Detection API
A FastAPI backend for detecting AI-generated voices from audio URLs.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional
import httpx
import librosa
import numpy as np
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
# Security (Swagger-compatible)
# --------------------------------------------------
security = HTTPBearer()

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    if token != VALID_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid authentication token")
    return token

# --------------------------------------------------
# Models
# --------------------------------------------------
class AudioRequest(BaseModel):
    audio_url: str = Field(
        ...,
        description="Public HTTP/HTTPS URL or local file URL (file:///...) to audio file"
    )

class PredictionResponse(BaseModel):
    classification: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    language: str
    explanation: str

# --------------------------------------------------
# Audio download
# --------------------------------------------------
async def download_audio(url: str) -> Path:
    try:
        async with httpx.AsyncClient(
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": "AI-Voice-Detection-Hackathon"}
        ) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download audio: HTTP {response.status_code}"
                    )

                extension = ".wav" if ".wav" in url.lower() else ".mp3"
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)

                total_size = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    total_size += len(chunk)
                    if total_size > MAX_AUDIO_SIZE_MB * 1024 * 1024:
                        temp_file.close()
                        os.unlink(temp_file.name)
                        raise HTTPException(
                            status_code=400,
                            detail="Audio file exceeds size limit"
                        )
                    temp_file.write(chunk)

                temp_file.close()
                return Path(temp_file.name)

    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Audio download timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
def extract_audio_features(audio_path: Path) -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=60)

        features = {}

        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

        features["spectral_centroid_mean"] = np.mean(sc)
        features["spectral_centroid_std"] = np.std(sc)
        features["spectral_bandwidth_mean"] = np.mean(sb)
        features["spectral_bandwidth_std"] = np.std(sb)

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zcr_mean"] = np.mean(zcr)
        features["zcr_std"] = np.std(zcr)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
            features[f"mfcc_{i}_std"] = np.std(mfccs[i])

        rms = librosa.feature.rms(y=y)[0]
        features["rms_mean"] = np.mean(rms)
        features["rms_std"] = np.std(rms)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = tempo
        features["duration"] = librosa.get_duration(y=y, sr=sr)

        return features

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process audio file: {str(e)}"
        )

# --------------------------------------------------
# Analysis
# --------------------------------------------------
def analyze_voice_characteristics(features: dict) -> dict:
    score = 0.5
    reasons = []

    sc_var = features["spectral_centroid_std"] / (features["spectral_centroid_mean"] + 1e-6)
    if sc_var < 0.15:
        score += 0.2
        reasons.append("highly consistent spectral properties")
    else:
        score -= 0.2
        reasons.append("natural spectral variation")

    rms_var = features["rms_std"] / (features["rms_mean"] + 1e-6)
    if rms_var < 0.4:
        score += 0.15
        reasons.append("uniform energy patterns")
    else:
        score -= 0.15
        reasons.append("natural dynamic range")

    confidence = float(np.clip(score, 0, 1))
    classification = "AI Generated" if confidence > 0.55 else "Human"

    explanation = f"Analysis suggests {classification.lower()} voice: {', '.join(reasons)}"
    if confidence < 0.2 or confidence > 0.8:
        explanation += " (high confidence)"

    return {
        "classification": classification,
        "confidence": round(confidence, 2),
        "explanation": explanation
    }

# --------------------------------------------------
# Language heuristic
# --------------------------------------------------
def detect_language(features: dict) -> str:
    tempo = features.get("tempo", 120)
    if tempo < 100:
        return "Spanish"
    if tempo > 130:
        return "English"
    return "English"

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
    parsed = urlparse(request.audio_url)

    try:
        if parsed.scheme == "file":
            audio_path = Path(parsed.path)
            if not audio_path.exists():
                raise HTTPException(status_code=400, detail="Local audio file not found")
        else:
            audio_path = await download_audio(request.audio_url)

        features = extract_audio_features(audio_path)
        analysis = analyze_voice_characteristics(features)
        language = detect_language(features)

        return PredictionResponse(
            classification=analysis["classification"],
            confidence=analysis["confidence"],
            language=language,
            explanation=analysis["explanation"]
        )

    finally:
        if audio_path and audio_path.exists() and parsed.scheme != "file":
            try:
                os.unlink(audio_path)
            except Exception:
                pass

# --------------------------------------------------
# Error handler
# --------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
