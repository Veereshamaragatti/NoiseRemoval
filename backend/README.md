# FastAPI Video Noise Removal Backend

## 🚀 Quick Start

### 1. Prerequisites

Ensure the virtual environment is set up:
- Virtual environment: `noise/` (already configured)
- Python 3.8+
- All dependencies installed (see main README.md)

### 2. Start the Server

**From the project root:**

```bash
# Windows PowerShell
.\noise\Scripts\Activate.ps1
cd backend
python app.py

# Linux/Mac
source noise/bin/activate
cd backend
python app.py
```

**Or using Uvicorn directly:**

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 3. Open the Frontend

Navigate to the project root and open `index.html` in your browser:

```bash
# Windows
start ..\index.html

# Linux
xdg-open ../index.html

# Mac
open ../index.html
```

---

## 📁 Backend File Structure

```
backend/
├── app.py              # FastAPI server (main entry point)
├── process_video.py    # AI video processing pipeline
├── __init__.py         # Python module initialization
├── README.md           # This file
├── __pycache__/        # Python bytecode cache (auto-generated)
├── uploads/            # Temporary storage for uploaded videos
│   └── .gitkeep        # Keeps folder in git
├── outputs/            # Processed videos ready for download
│   └── .gitkeep        # Keeps folder in git
├── subtitles/          # (Future feature) Subtitle files
└── transcripts/        # (Future feature) Transcript files
```

---

## 🔌 API Endpoints

### `GET /`
**Health check endpoint**

**Response:**
```json
{
  "status": "ok",
  "message": "AI Video Noise Removal API is running"
}
```

**Example:**
```bash
curl http://localhost:8000/
```

---

### `POST /upload`
**Upload and process a video file**

**Parameters:**
- `file` (form-data): Video file to process (MP4, AVI, MOV, MKV)
- `use_gpu` (form-data, optional): Enable GPU acceleration (default: false)
- `use_facebook_denoiser` (form-data, optional): Enable Facebook Denoiser (default: false)

**Response:**
```json
{
  "status": "done",
  "output_file": "/download/abc123_clean_synced.mp4",
  "stats": {
    "original_duration": 120.5,
    "processed_duration": 95.2,
    "silence_removed_percent": 21.0,
    "segments_removed": 8
  }
}
```

**Example:**
```bash
# Basic upload (CPU only)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_video.mp4"

# With GPU acceleration
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_video.mp4" \
  -F "use_gpu=true"

# With Facebook Denoiser (high quality)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_video.mp4" \
  -F "use_facebook_denoiser=true"

# GPU + Facebook Denoiser (best quality, fastest)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_video.mp4" \
  -F "use_gpu=true" \
  -F "use_facebook_denoiser=true"
```

---

### `GET /download/{filename}`
**Download a processed video file**

**Parameters:**
- `filename` (path): Name of the processed file

**Response:**
- Video file (MP4) with appropriate headers

**Example:**
```bash
curl -O "http://localhost:8000/download/abc123_clean_synced.mp4"
```

---

### `DELETE /cleanup/{filename}`
**Delete a processed video file**

**Parameters:**
- `filename` (path): Name of the file to delete

**Response:**
```json
{
  "status": "deleted",
  "filename": "abc123_clean_synced.mp4"
}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/cleanup/abc123_clean_synced.mp4"
```

---

## 🔧 Configuration

### Processing Options

Edit `process_video.py` to customize:

```python
# Audio sample rate (lower = faster, less quality)
"-ar", "16000"  # Default: 16kHz (change to 48000 for studio quality)

# Silence threshold
silence_thresh = dbfs - 16  # Adjust sensitivity

# Minimum silence length
min_silence_len = 300  # milliseconds
```

### Server Configuration

Edit `app.py` or use environment variables:

```python
# Change host/port
uvicorn.run("app:app", host="0.0.0.0", port=8000)

# Enable/disable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
)
```

---

## 🧪 Testing with Python

```python
import requests

# Upload a video
with open("test_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f},
        data={"use_gpu": "true"}
    )
    
result = response.json()
print(f"Processed: {result['output_file']}")
print(f"Stats: {result['stats']}")

# Download the result
download_url = f"http://localhost:8000{result['output_file']}"
output = requests.get(download_url)
with open("cleaned_video.mp4", "wb") as f:
    f.write(output.content)
```

---

## 📝 Processing Pipeline

The `process_video.py` module implements this pipeline:

```
1. Extract Audio (FFmpeg)
   └─> 16kHz mono WAV file

2. AI Denoising (DeepFilterNet)
   └─> Remove stationary noise (fans, AC, hum)

3. Optional: Facebook Denoiser
   └─> Remove non-stationary noise (complex backgrounds)

4. Speech Enhancement
   └─> Gating + EQ to boost voice clarity

5. Silence Detection (Pydub)
   └─> Find silent segments

6. Video Trimming (FFmpeg)
   └─> Remove silent parts from both audio & video

7. Final Merge
   └─> Combine trimmed video + cleaned audio

8. Cleanup
   └─> Delete temporary files
```

---

## ⚡ Performance Notes

### CPU Mode (Default)
- Works on all systems
- 1-minute video: ~2-4 minutes processing
- RAM: 2-3 GB

### GPU Mode (CUDA)
- Requires NVIDIA GPU
- 1-minute video: ~30-60 seconds processing
- VRAM: 2-3 GB

### Facebook Denoiser
- Higher quality, more memory
- Adds 50-100% processing time
- RAM: +4-5 GB

### Recommendations
- **Small videos (<5 min)**: CPU is fine
- **Large videos (>10 min)**: Use GPU
- **Studio quality**: Enable Facebook Denoiser
- **Quick preview**: CPU only, no Facebook Denoiser

---

## 🐛 Troubleshooting

### Issue: "FFmpeg not found"
**Solution:** The code uses bundled FFmpeg:
```python
FFMPEG_PATH = SCRIPT_DIR / "noise" / "third_party" / "ffmpeg" / "ffmpeg-8.0-essentials_build" / "bin" / "ffmpeg.exe"
```
Ensure this path exists or install FFmpeg system-wide.

### Issue: "CUDA out of memory"
**Solution:** 
- Use CPU mode
- Process shorter videos
- Disable Facebook Denoiser

### Issue: "Port 8000 already in use"
**Solution:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn app:app --port 8001
```

### Issue: "Module not found"
**Solution:**
```bash
# Ensure virtual environment is activated
.\noise\Scripts\Activate.ps1  # Windows
source noise/bin/activate      # Linux/Mac

# Reinstall dependencies
pip install -r ../requirements.txt
```

---

## 📊 API Response Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid file or parameters |
| 404 | Not Found | File doesn't exist |
| 500 | Server Error | Processing failed |

---

## 🔒 Security Notes

⚠️ **Current Implementation (Development Mode)**
- CORS: Allows all origins (`*`)
- No authentication
- No file size limits
- No rate limiting

🔐 **Production Recommendations**
- Restrict CORS to specific domains
- Add API authentication (JWT, API keys)
- Implement file size limits (max 500MB)
- Add rate limiting (max 10 requests/minute)
- Use HTTPS
- Validate file types strictly
- Implement user quotas

---

## 📦 Dependencies

Core libraries used:
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **DeepFilterNet** - AI noise removal
- **Facebook Denoiser** - Advanced denoising
- **PyTorch** - Deep learning framework
- **Librosa** - Audio analysis
- **Pydub** - Audio manipulation
- **FFmpeg** - Video/audio processing

See `requirements.txt` for complete list.

---

## 🚀 Deployment Options

### Local Development
```bash
uvicorn app:app --reload
```

### Production (Gunicorn + Uvicorn)
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Platforms
- **Heroku**: Add `Procfile` with `web: uvicorn app:app --host 0.0.0.0 --port $PORT`
- **AWS Lambda**: Use Mangum adapter
- **Google Cloud Run**: Use containerized deployment
- **Azure App Service**: Deploy as Python web app

---

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DeepFilterNet GitHub](https://github.com/Rikorose/DeepFilterNet)
- [Facebook Denoiser GitHub](https://github.com/facebookresearch/denoiser)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)

---

**For main project documentation, see the root `README.md` file.**
