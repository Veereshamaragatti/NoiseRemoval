# Video Translation & Transcription App with AI Noise Removal

A Flask-based web application that provides a **complete unified pipeline** for video processing:

1. **🔊 AI Noise Removal** - Remove background noise using DeepFilterNet + Facebook Denoiser
2. **✂️ Silence Removal** - Automatically detect and remove silence segments
3. **📝 Transcription** - Generate subtitles in the original language (auto-detected)
4. **🌍 Translation** - Translate audio to multiple languages with synchronized dubbed audio

## ✨ Key Features

### Unified Processing Pipeline
- **Single-click processing** - Noise removal + Silence removal + Transcription in one flow
- **Customizable options** - Choose which processing steps to apply
- **Perfect synchronization** - Audio and video stay perfectly in sync throughout
- **Clean output** - Processed video with clean audio and accurate subtitles

### AI-Powered Technologies
- **DeepFilterNet** - State-of-the-art noise reduction
- **Facebook Denoiser** - Advanced non-stationary noise removal
- **OpenAI Whisper** - Accurate speech recognition and transcription
- **Google Translate** - Multi-language translation
- **gTTS** - Natural text-to-speech synthesis

## Supported Languages

- 🇬🇧 English
- 🇮🇳 Hindi
- 🇮🇳 Kannada
- 🇮🇳 Telugu
- 🇮🇳 Tamil
- 🇮🇳 Malayalam
- 🇮🇳 Marathi
- 🇮🇳 Bengali
- 🇮🇳 Gujarati
- 🇮🇳 Punjabi
- 🇮🇳 Odia
- 🇮🇳 Urdu
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇰🇷 Korean

## Installation

### Prerequisites

1. **Python 3.8+** installed on your system
2. **FFmpeg** installed and added to PATH
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
3. **CUDA** (optional, for GPU acceleration - highly recommended for faster processing)

### Setup Steps

1. **Clone or download this project**

2. **Install Python dependencies**
   ```bash
   cd flask_app
   pip install -r requirements.txt
   ```
   
   **Note:** If you don't have GPU/CUDA, you can skip DeepFilterNet installation:
   ```bash
   pip install -r requirements.txt --no-deps denoiser
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   ```
   http://127.0.0.1:5000
   ```

## Usage Guide

### Option 1: Complete Processing (Recommended)

1. **Upload a Video**
   - Click "Choose a video file" on the homepage
   - Select any video file (MP4, AVI, MOV, etc.)
   - Click "Upload & Process"

2. **Process Video with Full Pipeline**
   - Click **"⚙️ Process Video (Noise + Silence + Transcribe)"**
   - Select processing options:
     - ✅ **Remove background noise** - Uses AI to clean audio
     - ✅ **Remove silence segments** - Cuts out dead air
     - ✅ **Generate transcription** - Creates subtitles
   - Click "Start Processing"
   - Wait for completion (may take 5-10 minutes for long videos)
   - Watch the processed video with clean audio and subtitles!

### Option 2: Translation

1. **After processing, translate the audio**
   - Click **"🌍 Translate Audio"**
   - Select target language
   - Click "Translate"
   - Watch with dubbed audio in the new language!

## Processing Pipeline Details

### Step 1: Noise Removal
- **DeepFilterNet** removes stationary background noise
- **Facebook Denoiser** handles non-stationary noise (traffic, crowds, etc.)
- **Speech-aware gating** preserves speech while reducing noise
- **EQ optimization** enhances voice clarity (1kHz-3kHz boost)

### Step 2: Silence Detection & Removal
- Detects silence segments using FFmpeg silencedetect
- Configurable threshold (-35dB) and duration (1 second)
- Precisely cuts silence from both video and audio
- Maintains perfect A/V synchronization

### Step 3: Transcription
- Extracts audio from processed video
- OpenAI Whisper performs speech-to-text
- Auto-detects language
- Generates WebVTT subtitles with timestamps

### Step 4: Translation (Optional)
- Translates each subtitle segment
- Generates TTS audio for each segment
- Time-stretches audio to match original timing
- Creates perfectly synced dubbed audio track

## Project Structure

```
flask_app/
│
├── app.py                    # Main Flask application
├── video_processor.py        # Unified video processing pipeline
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── templates/
│   ├── index.html           # Upload page
│   └── player.html          # Video player with processing controls
│
└── static/
    ├── uploads/             # Uploaded original videos
    └── processed/           # Processed videos and generated files
```

## Technologies Used

- **Flask** - Web framework
- **DeepFilterNet** - AI noise reduction
- **Facebook Denoiser** - Advanced denoising
- **OpenAI Whisper** - Speech recognition & transcription
- **Google Translate (deep-translator)** - Text translation
- **gTTS** - Text-to-speech synthesis
- **FFmpeg** - Audio/video processing
- **PyTorch** - Deep learning framework
- **Librosa** - Audio analysis
- **Pydub** - Audio manipulation

## Performance Notes

### Processing Time (for a 5-minute video)
- **Noise Removal**: ~2-3 minutes (GPU) / ~10-15 minutes (CPU)
- **Silence Removal**: ~30 seconds
- **Transcription**: ~1-2 minutes
- **Translation + TTS**: ~3-5 minutes per language

### GPU vs CPU
- **GPU (CUDA)**: 5-10x faster for noise removal
- **CPU**: Works but slower, suitable for short videos

### Model Downloads (First Run)
- Whisper "small" model: ~500MB
- DeepFilterNet: ~50MB
- Facebook Denoiser: ~100MB

## Troubleshooting

### FFmpeg not found
- Make sure FFmpeg is installed and in your system PATH
- Test by running `ffmpeg -version` in terminal

### CUDA/GPU Issues
- If you don't have NVIDIA GPU, the app will automatically use CPU
- For GPU: Install CUDA toolkit and PyTorch with CUDA support
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Noise Removal Models Not Loading
- If DeepFilterNet or Facebook Denoiser fail to install:
  - Processing will continue without noise removal
  - Or install manually: `pip install DeepFilterNet denoiser`

### Whisper Model Loading Slow
- First run downloads the model (500MB-1.5GB depending on size)
- Subsequent runs will be faster
- Change model size in `app.py`: "base" (fastest) to "large" (most accurate)

### Translation Errors
- Check internet connection (required for Google Translate)
- Some languages may have limited TTS support

### Audio Sync Issues
- The app uses advanced time-stretching for perfect sync
- Long videos may take more time to process
- GPU acceleration significantly improves speed

### Memory Issues
- Long videos (>30 min) may require 8GB+ RAM
- Consider splitting long videos into smaller segments
- Use smaller Whisper model ("base" or "small")

## Configuration

Edit `app.py` to customize:

```python
# Whisper model size (base/small/medium/large)
WHISPER_MODEL_NAME = "small"

# Silence detection threshold
noise_threshold = "-35dB"  # Lower = more sensitive
min_silence_duration = 1.0  # seconds
```

Edit `video_processor.py` for advanced settings:

```python
# Audio sample rate
sample_rate = 48000

# Speech enhancement EQ range
eq_range = [1000, 3000]  # Hz

# Noise gate threshold
gate_threshold = 0.8  # relative to median energy
```

## Notes

- Processing time depends on video length and system performance
- First run will download Whisper model (~500MB for "small" model)
- Generated files are stored in `static/uploads/`
- You can change Whisper model size in `app.py` (base/small/medium/large)

## License

This project is open source and available for educational purposes.

## Credits

Built using OpenAI Whisper, Google Translate API, and gTTS.
