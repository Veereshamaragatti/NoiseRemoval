# Quick Server Restart Instructions

## The Problem
The uvicorn server with --reload didn't pick up all the changes we made to app.py

## The Solution

### Step 1: Stop the Current Server
In your terminal where uvicorn is running, press:
```
Ctrl + C
```

### Step 2: Restart the Server
```powershell
python -m uvicorn app:app --reload
```

### Step 3: Open Browser
Navigate to one of these URLs:
- http://localhost:8000/
- http://localhost:8000/index_merged.html

## Expected Output When Server Starts

You should see:
```
🚀 Starting FastAPI server...
📁 Upload directory: e:\NoiseRemoval\backend\uploads
📁 Output directory: e:\NoiseRemoval\backend\outputs
🌐 Open frontend at: http://localhost:8000/
🌐 Or directly: http://localhost:8000/index_merged.html
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## If It Still Doesn't Work

1. Check the terminal for any error messages
2. Verify you're in the correct directory: `E:\NoiseRemoval\backend`
3. Try accessing: http://127.0.0.1:8000/index_merged.html (use 127.0.0.1 instead of localhost)

## Troubleshooting Commands

Test if server is running:
```powershell
curl http://localhost:8000/api/status
```

Should return:
```json
{
  "status": "ok",
  "message": "AI Video Processing API is running",
  "features": ["noise_removal", "silence_trimming", "transcription", "keyword_search"]
}
```
