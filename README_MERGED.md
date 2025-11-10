# AI Video Processing - Merged Project

## 🎯 Overview

This is a comprehensive video processing application that combines **AI-powered noise removal** with **multi-language speech transcription and keyword search**. The project merges two separate implementations:

1. **Noise Removal & Silence Trimming** - AI-based audio enhancement using DeepFilterNet and Facebook Denoiser
2. **Multi-language Transcription & Keyword Search** - Speech-to-text in 12 Indian languages with searchable subtitles

## ✨ Features

### 🔊 Audio Enhancement
- **AI Noise Removal** - DeepFilterNet for stationary noise reduction
- **Facebook Denoiser** - Optional enhanced denoising for non-stationary noise
- **Silence Detection & Removal** - Automatically trim silent portions
- **Speech-aware Audio Gating** - Preserves speech quality while removing background noise
- **Dynamic EQ** - Frequency enhancement for clearer speech
- **Audio Normalization** - Consistent volume levels
- **GPU Acceleration** - Optional CUDA support for faster processing

### 🎙️ Transcription & Search
- **Multi-language Support** - 12 Indian languages + English:
  - English, Hindi, Kannada, Telugu, Tamil
  - Malayalam, Marathi, Gujarati, Bengali
  - Punjabi, Odia, Urdu
- **Automatic Transcription** - Using OpenAI Whisper
- **Auto-translation** - Google Translate integration for multi-language subtitles
- **VTT Subtitle Generation** - WebVTT format for video players
- **Keyword Search** - Find words in transcripts and jump to timestamps
- **Search Index** - Fast word-to-timestamp mapping

### 🎬 Video Processing
- **Format Support** - MP4, AVI, MOV, MKV, WebM
- **Audio Extraction** - Automatic audio track processing
- **Video Synchronization** - Perfect A/V sync after editing
- **Segment Concatenation** - Seamless silence removal
- **Side-by-side Comparison** - Original vs processed video

## 🏗️ Architecture

```
NoiseRemoval/
├── backend/
│   ├── app.py                 # Main FastAPI application
│   ├── process_video.py       # Video processing pipeline
│   ├── transcribe.py          # Whisper transcription module
│   ├── search_index.py        # Keyword search indexing
│   ├── vtt_utils.py          # VTT file parser
│   ├── uploads/              # Temporary upload storage
│   ├── outputs/              # Processed video files
│   ├── subtitles/            # Generated VTT files & indexes
│   └── transcripts/          # Additional transcript storage
├── noise/                     # Virtual environment
│   └── third_party/
│       └── ffmpeg/           # FFmpeg binaries
├── index_merged.html         # Enhanced frontend (NEW)
├── index.html                # Original noise removal frontend
├── requirements.txt          # Python dependencies
└── README_MERGED.md          # This file
```

## 🚀 Setup Instructions

### Prerequisites

1. **Python 3.9+**
2. **FFmpeg** - Already included in `noise/third_party/ffmpeg/`
3. **CUDA Toolkit** (Optional) - For GPU acceleration

### Installation

1. **Activate Virtual Environment**
   ```powershell
   cd e:\NoiseRemoval
   .\noise\Scripts\Activate.ps1
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

   This will install:
   - `torch`, `torchaudio` - Deep learning framework
   - `deepfilternet` - AI noise removal
   - `denoiser` - Facebook's denoiser
   - `openai-whisper` - Speech-to-text
   - `deep-translator` - Multi-language translation
   - `librosa`, `soundfile`, `pydub` - Audio processing
   - `fastapi`, `uvicorn` - Web framework
   - `scipy`, `numpy` - Scientific computing

3. **Configure Python Environment** (if needed)
   ```powershell
   # The transcribe.py module will auto-detect FFmpeg
   # Verify FFmpeg is accessible
   .\noise\third_party\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe -version
   ```

## 🎮 Usage

### Start the Backend

```powershell
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Access the Frontend

Open in browser: `http://localhost:8000/index_merged.html`

Or use the original frontend: `http://localhost:8000/../index.html`

### Processing Workflow

1. **Upload Video/Audio**
   - Click "Choose Video/Audio File"
   - Select your media file

2. **Configure Options**
   - ✅ **Use GPU Acceleration** - Enable if you have CUDA GPU
   - ✅ **Facebook Denoiser** - Higher quality, more memory
   - ✅ **Enable Transcription** - Generate searchable subtitles
   - Select languages for transcription (default: all languages)

3. **Process**
   - Click "Upload & Process Video"
   - Wait for processing (may take several minutes)

4. **Review Results**
   - View processing statistics (duration, silence removed, etc.)
   - Compare original vs processed video side-by-side
   - Download processed video

5. **Search Transcripts** (if transcription enabled)
   - Type keyword in search box
   - Select language
   - Click "Search" to jump to timestamps
   - Click time chips to navigate to specific occurrences

## 📡 API Endpoints

### `GET /`
Health check endpoint
```json
{
  "status": "ok",
  "message": "AI Video Processing API is running",
  "features": ["noise_removal", "silence_trimming", "transcription", "keyword_search"]
}
```

### `GET /api/langs`
Get supported languages for transcription
```json
{
  "en": "English",
  "hi": "Hindi",
  "kn": "Kannada",
  ...
}
```

### `POST /upload`
Upload and process video

**Form Data:**
- `file` - Video/audio file
- `use_gpu` - Boolean (default: false)
- `use_facebook_denoiser` - Boolean (default: false)
- `enable_transcription` - Boolean (default: false)
- `transcription_langs` - Comma-separated language codes (e.g., "en,hi,ta")

**Response:**
```json
{
  "status": "done",
  "video_id": "abcd1234",
  "output_file": "/download/abcd1234_clean_synced.mp4",
  "original_file": "/download/abcd1234_original.mp4",
  "statistics": {
    "original_duration": 120.5,
    "processed_duration": 95.3,
    "silence_removed_percent": 20.9,
    "segments_removed": 15
  },
  "transcription": {
    "enabled": true,
    "languages": ["en", "hi"],
    "tracks": [
      {
        "lang": "en",
        "label": "English",
        "url": "/subtitles/abcd1234.en.vtt"
      }
    ]
  }
}
```

### `GET /api/search`
Search for keywords in transcripts

**Query Parameters:**
- `video_id` - Video identifier
- `q` - Search keyword
- `lang` - Language code (default: "en")

**Response:**
```json
{
  "video_id": "abcd1234",
  "lang": "en",
  "keyword": "hello",
  "hits": [12.5, 45.3, 78.9],
  "count": 3
}
```

### `GET /download/{filename}`
Download processed video file

### `DELETE /cleanup/{filename}`
Delete processed video file

## ⚙️ Configuration

### Environment Variables

```powershell
# Set Whisper model size (optional)
$env:WHISPER_MODEL = "small"  # Options: tiny, base, small, medium, large

# For GPU processing, ensure CUDA is installed
# The app will auto-detect CUDA availability
```

### Model Sizes

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| tiny  | 75 MB | Very Fast | Basic |
| base  | 142 MB | Fast | Good |
| small | 466 MB | Medium | Better |
| medium | 1.5 GB | Slow | Very Good |
| large | 2.9 GB | Very Slow | Best |

## 🔧 How It Works

### Processing Pipeline

```
Input Video
    ↓
1. Extract Audio (16kHz mono)
    ↓
2. DeepFilterNet (Stationary Noise Removal)
    ↓
3. Facebook Denoiser (Optional - Non-stationary Noise)
    ↓
4. Speech-aware Gating + EQ Enhancement
    ↓
5. Audio Normalization
    ↓
6. Silence Detection & Segmentation
    ↓
7. Video/Audio Trimming & Concatenation
    ↓
8. A/V Sync & Final Merge
    ↓
9. Whisper Transcription (Optional)
    ↓
10. Multi-language Translation (Optional)
    ↓
11. VTT Generation + Search Index
    ↓
Final Output + Subtitles
```

### Transcription Pipeline

```
Processed Audio
    ↓
Whisper Model (English transcription)
    ↓
Google Translate (Other languages)
    ↓
VTT File Generation (per language)
    ↓
Word Tokenization & Indexing
    ↓
Search Index (JSON)
```

## 🐛 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Check: `noise/third_party/ffmpeg/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe` exists
   - The app should auto-configure FFmpeg path

2. **CUDA not available**
   - GPU processing will fall back to CPU automatically
   - Install PyTorch with CUDA for GPU support
   - Check: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Translation fails**
   - Requires internet connection for Google Translate
   - Falls back to English transcription if translation fails
   - Install: `pip install deep-translator`

4. **Out of memory**
   - Disable Facebook Denoiser (uses more memory)
   - Use smaller Whisper model (tiny or base)
   - Process shorter video segments

5. **Slow processing**
   - Enable GPU acceleration if available
   - Use smaller Whisper model
   - Reduce number of transcription languages

## 📊 Performance Tips

- **GPU vs CPU**: GPU is 5-10x faster for transcription
- **Model Selection**: `small` model offers best speed/quality balance
- **Batch Processing**: Process multiple videos sequentially
- **Language Selection**: Only select needed languages to save time
- **Facebook Denoiser**: Skip if you don't need maximum quality

## 🤝 Credits

### Original Projects

1. **Noise Removal Project** - Veeresh Amaragatti
2. **Transcription Project** - Bindu (https://github.com/Bindugowda2004)

### Technologies Used

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) - Noise suppression
- [Facebook Denoiser](https://github.com/facebookresearch/denoiser) - Speech enhancement
- [Deep Translator](https://github.com/nidhaloff/deep-translator) - Multi-language translation
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [FFmpeg](https://ffmpeg.org/) - Media processing

## 📄 License

MIT License - Use freely with attribution

## 🎓 Academic Use

This merged project is suitable for:
- Capstone projects
- Research demonstrations
- Educational purposes
- Portfolio projects

## 📞 Support

For issues or questions:
1. Check this README
2. Review API documentation
3. Inspect browser console for errors
4. Check backend terminal output

## 🚧 Future Enhancements

- [ ] Real-time processing preview
- [ ] Batch video processing
- [ ] Custom noise profiles
- [ ] Export subtitle files separately
- [ ] Advanced search (regex, phrases)
- [ ] Video annotation tools
- [ ] Cloud deployment guide
- [ ] Docker containerization
- [ ] Mobile-responsive UI improvements
