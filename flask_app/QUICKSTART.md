# Quick Start Guide

## Installation (Windows)

1. **Install Python 3.8+** from python.org

2. **Install FFmpeg**
   - Download from: https://ffmpeg.org/download.html
   - Extract to `C:\ffmpeg`
   - Add `C:\ffmpeg\bin` to system PATH

3. **Install dependencies**
   ```powershell
   cd E:\NoiseRemoval\flask_app
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```powershell
   python app.py
   ```

5. **Open browser**
   - Go to: http://127.0.0.1:5000

## Quick Test

1. Upload a video
2. Click "Process Video (Noise + Silence + Transcribe)"
3. Check all options
4. Click "Start Processing"
5. Wait for completion
6. Watch the cleaned video with subtitles!

## What Gets Processed?

✅ **Background noise removed** - AI cleans the audio
✅ **Silence removed** - Dead air is cut out
✅ **Subtitles generated** - Automatic transcription
✅ **Video synced** - Perfect audio-video synchronization

## Optional: Translation

After processing, click "Translate Audio" to:
- Translate subtitles to another language
- Generate dubbed audio in that language
- Watch with synchronized dubbing

## Tips

- **First run**: Downloads AI models (~1GB), takes longer
- **GPU recommended**: 5-10x faster processing
- **Short videos first**: Test with 1-2 minute videos
- **Processing time**: ~2-5 minutes per minute of video (GPU)

## Common Issues

**"FFmpeg not found"**
- Add ffmpeg to PATH and restart terminal

**"CUDA not available"**
- App will use CPU (slower but works)
- For GPU: Install CUDA toolkit

**Processing takes long**
- Normal for CPU processing
- First run downloads models
- GPU speeds up 5-10x

**Out of memory**
- Use shorter videos
- Close other applications
- Use smaller Whisper model ("base")

## Need Help?

Check README.md for detailed documentation!
