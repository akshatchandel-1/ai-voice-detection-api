# AI-Generated Voice Detection API

A production-ready FastAPI backend for detecting AI-generated voices from audio files. Built for hackathon submission with stability, reliability, and ease of deployment in mind.

## Features

- ✅ **Single POST endpoint** (`/predict`) for voice classification
- ✅ **Audio URL processing** - Downloads and analyzes MP3/WAV files from public URLs
- ✅ **Bearer token authentication** - Simple but secure auth header validation
- ✅ **Real audio analysis** - Uses librosa for feature extraction and heuristic ML-based detection
- ✅ **Structured JSON responses** - Classification, confidence, language, and explanation
- ✅ **Production-ready** - Error handling, logging, file cleanup, and size limits
- ✅ **Easy deployment** - Works on Render, Railway, Fly.io, or any Python hosting platform

## Technology Stack

- **FastAPI** - Modern, fast web framework
- **Librosa** - Audio analysis and feature extraction
- **Pydantic** - Data validation and serialization
- **HTTPX** - Async HTTP client for audio downloads
- **NumPy** - Numerical computations

## Quick Start

### Local Development

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the server**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

3. **View API documentation**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Deploy to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy!

## API Endpoints

### `POST /predict`

Analyzes an audio file and detects whether it's AI-generated or human voice.

**Authentication**: Requires `Authorization` header with Bearer token

**Request Body**:
```json
{
  "audio_url": "https://example.com/audio-file.mp3"
}
```

**Response**:
```json
{
  "classification": "AI Generated",
  "confidence": 0.87,
  "language": "English",
  "explanation": "Analysis suggests AI generation: highly consistent spectral properties, uniform voice activity patterns, unnatural prosody patterns (high confidence)"
}
```

### `GET /health`

Health check endpoint to verify service status.

**Response**:
```json
{
  "status": "healthy",
  "service": "ai-voice-detection",
  "checks": {
    "api": "ok",
    "audio_processing": "ok"
  }
}
```

## API Usage Examples

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer hackathon_2024_secret_token" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
  }'
```

### Using Python (requests)

```python
import requests

url = "http://localhost:8000/predict"
headers = {
    "Authorization": "Bearer hackathon_2024_secret_token",
    "Content-Type": "application/json"
}
data = {
    "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### Using JavaScript (fetch)

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer hackathon_2024_secret_token',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    audio_url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3'
  })
});

const result = await response.json();
console.log(result);
```

### Using Postman

1. Create a new POST request to `http://localhost:8000/predict`
2. Add header: `Authorization: Bearer hackathon_2024_secret_token`
3. Set body type to JSON (raw)
4. Add JSON body:
```json
{
  "audio_url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"
}
```
5. Send request

## Authentication

The API uses Bearer token authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer hackathon_2024_secret_token
```

**Default token**: `hackathon_2024_secret_token`

To change the token, edit the `VALID_BEARER_TOKEN` constant in `main.py`.

## Response Format

All successful predictions return a JSON object with:

| Field | Type | Description |
|-------|------|-------------|
| `classification` | string | Either "AI Generated" or "Human" |
| `confidence` | float | Confidence score between 0.0 and 1.0 |
| `language` | string | Detected or assumed language (e.g., "English") |
| `explanation` | string | Human-readable explanation of the classification |

## Error Responses

The API returns structured error responses:

**401 Unauthorized** - Missing or invalid authentication
```json
{
  "error": "Authorization header missing",
  "status_code": 401
}
```

**400 Bad Request** - Invalid audio URL or processing error
```json
{
  "error": "Failed to download audio: HTTP 404",
  "status_code": 400
}
```

**408 Request Timeout** - Audio download timed out
```json
{
  "error": "Audio download timed out",
  "status_code": 408
}
```

**500 Internal Server Error** - Unexpected server error
```json
{
  "error": "Internal server error during audio processing",
  "status_code": 500
}
```

## How It Works

### Audio Processing Pipeline

1. **Download**: Audio file is downloaded from the provided URL (max 50MB, 30s timeout)
2. **Feature Extraction**: Librosa extracts 40+ audio features including:
   - Spectral features (centroid, rolloff, bandwidth)
   - MFCCs (Mel-frequency cepstral coefficients)
   - Zero-crossing rate
   - Chroma features
   - RMS energy
   - Tempo
3. **Analysis**: Heuristic analysis examines:
   - Spectral consistency (AI voices are more uniform)
   - Prosody variation (human voices have natural variation)
   - Energy distribution patterns
   - Frequency range characteristics
4. **Classification**: Based on multiple signals, returns classification with confidence score

### Detection Heuristics

The API uses several indicators of AI-generated speech:

- **Spectral Consistency**: AI voices often maintain more consistent spectral properties
- **Prosody Patterns**: Natural human speech has more variation in pitch and rhythm
- **Energy Distribution**: AI-generated audio can show uniform energy levels
- **Frequency Range**: Limited variation in spectral bandwidth suggests synthesis

## Configuration

Key constants in `main.py`:

```python
VALID_BEARER_TOKEN = "hackathon_2024_secret_token"  # Auth token
MAX_AUDIO_SIZE_MB = 50  # Maximum audio file size
TIMEOUT_SECONDS = 30     # Download timeout
```

## Testing with Public Audio URLs

Test audio files you can use:

- **Music (likely classified as AI)**: https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3
- **Speech samples**: Search for public domain speech recordings
- **Sample generator**: https://www.sample-videos.com/audio/

## Limitations

- Language detection uses simplified heuristics (for production, use a proper language ID model)
- Classification is based on audio characteristics, not perfect ML model
- Limited to 50MB audio files and 60-second processing
- Requires publicly accessible audio URLs

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest httpx pytest-asyncio

# Run tests (create test_main.py)
pytest test_main.py -v
```

### Code Structure

```
.
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Deployment Checklist

- [ ] Update `VALID_BEARER_TOKEN` for production
- [ ] Set appropriate `MAX_AUDIO_SIZE_MB` for your use case
- [ ] Configure logging level in production
- [ ] Set up monitoring and error tracking
- [ ] Add rate limiting if needed
- [ ] Configure CORS if serving frontend

## License

MIT License - Free to use for hackathons and projects

## Support

For issues or questions, check the FastAPI documentation at https://fastapi.tiangolo.com/
