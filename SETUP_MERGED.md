# Quick Setup Guide - Merged Video Processing Project

## ⚡ Quick Start (3 Steps)

### 1. Activate Environment
```powershell
cd e:\NoiseRemoval
.\noise\Scripts\Activate.ps1
```

### 2. Install New Dependencies
```powershell
pip install openai-whisper deep-translator
```

### 3. Start Server
```powershell
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open Browser
Navigate to: `http://localhost:8000/index_merged.html`

---

## 📋 What Changed?

### New Files Added
- ✅ `backend/transcribe.py` - Whisper transcription module
- ✅ `backend/search_index.py` - Keyword search engine
- ✅ `backend/vtt_utils.py` - Subtitle file parser
- ✅ `index_merged.html` - Enhanced frontend with all features
- ✅ `README_MERGED.md` - Complete documentation

### Modified Files
- ✅ `backend/app.py` - Added transcription & search endpoints
- ✅ `backend/process_video.py` - Added transcription pipeline
- ✅ `requirements.txt` - Added whisper & translation libraries

### New Directories (Auto-created)
- 📁 `backend/subtitles/` - VTT files and search indexes
- 📁 `backend/transcripts/` - Additional transcript storage

---

## 🎯 Key Features

### Noise Removal (Original)
- ✨ AI-powered noise reduction
- ✨ Silence detection & trimming
- ✨ GPU acceleration support
- ✨ Audio normalization

### Transcription (New)
- 🎙️ Multi-language speech-to-text
- 🌐 12 Indian languages + English
- 📝 VTT subtitle generation
- 🔍 Keyword search with timestamps

---

## 🧪 Testing the Integration

### Test Scenario 1: Noise Removal Only
1. Upload video
2. Enable GPU (optional)
3. Enable Facebook Denoiser (optional)
4. **Disable** transcription
5. Click "Upload & Process"
6. Result: Clean video without subtitles

### Test Scenario 2: Full Pipeline
1. Upload video
2. Enable transcription
3. Select languages (e.g., English, Hindi)
4. Click "Upload & Process"
5. Result: Clean video + searchable subtitles

### Test Scenario 3: Keyword Search
1. After processing with transcription
2. Type keyword (e.g., "hello", "नमस्ते")
3. Select language
4. Click "Search"
5. Result: Jump to timestamp, see all occurrences

---

## 🔧 API Testing

### Upload with Transcription
```powershell
# Using curl (if available)
curl -X POST http://localhost:8000/upload `
  -F "file=@C:\path\to\video.mp4" `
  -F "use_gpu=false" `
  -F "use_facebook_denoiser=false" `
  -F "enable_transcription=true" `
  -F "transcription_langs=en,hi"
```

### Search Transcripts
```powershell
curl "http://localhost:8000/api/search?video_id=abcd1234&lang=en&q=hello"
```

### Get Supported Languages
```powershell
curl http://localhost:8000/api/langs
```

---

## 📊 Expected Processing Times

| Video Length | CPU Only | GPU (CUDA) |
|--------------|----------|------------|
| 1 minute     | ~2-3 min | ~30-45 sec |
| 5 minutes    | ~10-15 min | ~2-3 min |
| 10 minutes   | ~20-30 min | ~4-6 min |

*Times include noise removal + transcription*

---

## 🐛 Quick Troubleshooting

### "Module not found: whisper"
```powershell
pip install openai-whisper
```

### "Module not found: deep_translator"
```powershell
pip install deep-translator
```

### Transcription takes too long
- Use smaller Whisper model: `$env:WHISPER_MODEL = "tiny"`
- Reduce number of languages
- Enable GPU if available

### Search returns empty results
- Ensure transcription was enabled during upload
- Check that subtitle files exist in `backend/subtitles/`
- Verify keyword matches the language selected

---

## 🎬 Demo Workflow

1. **Prepare test video**: Any MP4 with speech (30sec - 2min recommended)

2. **Start backend**:
   ```powershell
   cd backend
   python -m uvicorn app:app --reload
   ```

3. **Open frontend**: `http://localhost:8000/index_merged.html`

4. **Process video**:
   - Upload your test video
   - Enable all options for full demo
   - Select 2-3 languages (e.g., English, Hindi, Kannada)
   - Click "Upload & Process"

5. **Compare results**:
   - Listen to noise reduction quality
   - Check silence removal statistics
   - Enable subtitles on processed video

6. **Test search**:
   - Type a word from the speech
   - See timestamp highlights
   - Click time chips to jump

---

## 🔐 Security Notes

- **CORS enabled** for local development (`allow_origins=["*"]`)
- For production, restrict CORS origins
- Add authentication for upload endpoint
- Implement rate limiting for API calls
- Sanitize file uploads (already basic validation exists)

---

## 📦 Production Deployment

### Recommended Changes

1. **Update CORS**:
   ```python
   # In backend/app.py
   allow_origins=["https://yourdomain.com"]
   ```

2. **Add file size limits**:
   ```python
   # In backend/app.py
   from fastapi import File, UploadFile
   async def upload_video(file: UploadFile = File(..., max_length=500_000_000)):  # 500MB
   ```

3. **Environment variables**:
   ```powershell
   $env:WHISPER_MODEL = "base"  # Smaller for production
   $env:MAX_UPLOAD_SIZE = "500000000"
   $env:ENABLE_GPU = "true"
   ```

4. **Run with gunicorn** (Linux) or **waitress** (Windows):
   ```powershell
   pip install waitress
   waitress-serve --host=0.0.0.0 --port=8000 backend.app:app
   ```

---

## 📞 Need Help?

1. Check `README_MERGED.md` for detailed documentation
2. Review backend terminal for error messages
3. Check browser console (F12) for frontend errors
4. Verify all dependencies installed: `pip list`

---

## ✅ Success Checklist

- [ ] Virtual environment activated
- [ ] New dependencies installed
- [ ] Backend starts without errors
- [ ] Frontend loads in browser
- [ ] Can upload and process video (noise removal)
- [ ] Transcription generates VTT files
- [ ] Keyword search returns results
- [ ] Subtitles display in video player
- [ ] Download links work

---

## 🎉 You're Ready!

The merged project combines the best of both implementations:
- **Your work**: Advanced noise removal & silence trimming
- **Friend's work**: Multi-language transcription & search

Together: A complete video processing pipeline! 🚀
