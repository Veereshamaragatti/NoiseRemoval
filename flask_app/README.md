# Video Translation & Transcription App

A Flask-based web application that allows users to upload videos, transcribe audio in the original language, and translate audio into multiple languages with synchronized dubbed audio.

## Features

1. **Upload Video** - Upload any video file with audio
2. **Transcribe Audio** - Generate subtitles in the original language (auto-detected)
3. **Translate Audio** - Translate to multiple languages with synchronized audio dubbing
4. **Watch with Subtitles** - View videos with original or translated subtitles
5. **Audio Dubbing** - Listen to AI-generated dubbed audio that syncs perfectly with video

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

### Setup Steps

1. **Clone or download this project**

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
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

### 1. Upload a Video
- Click "Choose a video file" on the homepage
- Select any video file (MP4, AVI, MOV, etc.)
- Click "Upload & Process"

### 2. Transcribe Audio (Original Language)
- Click the **"📝 Transcribe Audio"** button
- Wait for processing (may take a few minutes for long videos)
- Original language subtitles will be added to the video
- The detected language will be displayed

### 3. Translate Audio
- Click the **"🌍 Translate Audio"** button
- Select target language from the dropdown
- Click "Translate"
- Wait for processing (includes translation + TTS generation)
- Translated subtitles and dubbed audio will be added

### 4. Watch & Listen
- Play the video to see subtitles
- When translated audio is available, it plays automatically (video audio is muted)
- Subtitles and audio stay in perfect sync

## How It Works

### Transcription
1. Audio is extracted from video using FFmpeg
2. OpenAI Whisper performs speech-to-text transcription
3. Language is auto-detected
4. Subtitles (VTT format) are generated with timestamps

### Translation
1. Video audio is transcribed (same as above)
2. Each subtitle segment is translated using Google Translate
3. Text-to-Speech (gTTS) generates audio for each segment
4. Audio segments are time-stretched to match original timing
5. Final dubbed audio is synced perfectly with video timeline

## Project Structure

```
flask_app/
│
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── templates/
│   ├── index.html           # Upload page
│   └── player.html          # Video player with controls
│
└── static/
    └── uploads/             # Uploaded videos and generated files
```

## Technologies Used

- **Flask** - Web framework
- **OpenAI Whisper** - Speech recognition & transcription
- **Google Translate (deep-translator)** - Text translation
- **gTTS** - Text-to-speech synthesis
- **FFmpeg** - Audio/video processing
- **Pydub** - Audio manipulation
- **MoviePy** - Video editing

## Troubleshooting

### FFmpeg not found
- Make sure FFmpeg is installed and in your system PATH
- Test by running `ffmpeg -version` in terminal

### Whisper model loading slow
- First run downloads the model (can be 1-2 GB)
- Subsequent runs will be faster

### Translation errors
- Check internet connection (required for Google Translate)
- Some languages may have limited TTS support

### Audio sync issues
- The app uses advanced time-stretching for perfect sync
- Long videos may take more time to process

## Notes

- Processing time depends on video length and system performance
- First run will download Whisper model (~500MB for "small" model)
- Generated files are stored in `static/uploads/`
- You can change Whisper model size in `app.py` (base/small/medium/large)

## License

This project is open source and available for educational purposes.

## Credits

Built using OpenAI Whisper, Google Translate API, and gTTS.
