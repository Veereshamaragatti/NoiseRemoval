# 🎬 AI Video Noise Removal System

> **Professional AI-powered video noise removal and silence trimming tool with a modern web interface**

This project uses cutting-edge AI models (DeepFilterNet3 & Facebook Denoiser) to automatically remove background noise from videos and trim silent segments while keeping perfect audio-video sync.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.119.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Configuration Options](#-configuration-options)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Performance Tips](#-performance-tips)
- [Credits](#-credits)

---

## ✨ Features

### 🎯 **Core Capabilities**
- ✅ **AI-Powered Noise Removal** - Uses DeepFilterNet3 neural network
- ✅ **Optional Enhanced Denoising** - Facebook Denoiser for superior quality
- ✅ **Automatic Silence Detection & Removal** - Trims dead air automatically
- ✅ **Perfect Audio-Video Sync** - Maintains perfect synchronization
- ✅ **GPU Acceleration** - CUDA support for faster processing
- ✅ **Before/After Comparison** - Side-by-side video comparison
- ✅ **Processing Statistics** - Shows silence removed, time saved, etc.

### 🎨 **User Interface**
- 🌐 **Modern Web Interface** - Beautiful, responsive design
- 📊 **Real-time Statistics** - Duration, silence %, time saved
- 🎬 **Video Comparison** - Play original vs cleaned side-by-side
- 📱 **Mobile Friendly** - Works on phones and tablets
- 🔄 **Progress Indicators** - Visual feedback during processing

### ⚙️ **Technical Features**
- 🚀 **FastAPI Backend** - High-performance async API
- 🔧 **Configurable Processing** - CPU/GPU toggle, quality options
- 📦 **Automatic Cleanup** - Removes temporary files
- 🎯 **Multiple Format Support** - MP4, AVI, MOV, MKV

---

## 🎥 Demo

**Processing Flow:**
1. Upload video → 2. AI removes noise → 3. Trims silence → 4. Compare & download

**Example Results:**
- Original: 120 seconds with background noise
- Processed: 95 seconds, crystal clear audio
- Result: 20% silence removed, professional quality

---

## 💻 System Requirements

### **Minimum Requirements (CPU Mode)**
- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

### **Recommended (GPU Mode)**
- **GPU**: NVIDIA GPU with CUDA support (GTX 1060 or better)
- **CUDA**: CUDA Toolkit 11.8 or compatible
- **RAM**: 16GB+ (for Facebook Denoiser)
- **VRAM**: 6GB+ GPU memory

### **Software Prerequisites**
- Python 3.8+
- FFmpeg (included in project)
- Modern web browser (Chrome, Edge, Firefox)

---

## 📦 Installation

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/Veereshamaragatti/NoiseRemoval.git
cd NoiseRemoval
```

### **Step 2: Set Up Python Virtual Environment**

**Windows (PowerShell):**
```powershell
python -m venv noise
.\noise\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python3 -m venv noise
source noise/bin/activate
```

### **Step 3: Install Core Dependencies**

```bash
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Note**: For CPU-only (no CUDA):
> ```bash
> pip install torch torchaudio
> ```

### **Step 4: Install AI Models & Libraries**

```bash
# DeepFilterNet (Primary noise removal)
pip install deepfilternet

# Facebook Denoiser (Advanced noise removal)
pip install denoiser

# Audio processing libraries
pip install librosa soundfile pydub scipy numpy

# API & Web Framework
pip install fastapi uvicorn python-multipart
```

### **Step 5: Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import df; print('DeepFilterNet: OK')"
python -c "import denoiser; print('Facebook Denoiser: OK')"
```

**Expected Output:**
```
PyTorch: 2.x.x+cu118
CUDA Available: True  # or False for CPU mode
DeepFilterNet: OK
Facebook Denoiser: OK
```

---

## 🚀 Quick Start

### **1. Start the Backend Server**

```bash
cd backend
python app.py
```

**Expected Output:**
```
🚀 Starting FastAPI server...
📁 Upload directory: E:\NoiseRemoval\backend\uploads
📁 Output directory: E:\NoiseRemoval\backend\outputs
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### **2. Open the Web Interface**

**Option A: Double-click** `index.html` in File Explorer

**Option B: Command line:**
```bash
# Windows
start index.html

# Linux
xdg-open index.html

# Mac
open index.html
```

### **3. Process Your First Video**

1. Click **"Choose Video File"**
2. Select a video (MP4, AVI, MOV, MKV)
3. Choose options:
   - ☐ **Use GPU** - Enable if you have NVIDIA GPU
   - ☐ **Use Facebook Denoiser** - Enable for best quality (needs 16GB RAM)
4. Click **"Upload & Process Video"**
5. Wait for processing (may take 2-10 minutes)
6. Compare before/after videos
7. Download the cleaned video

---

## 📁 Project Structure

```
NoiseRemoval/
├── 📄 index.html                    # Frontend web interface
├── 📄 deepfilternet_denoise.py      # Original standalone script
├── 📄 README.md                     # This file
│
├── 📁 backend/                      # FastAPI backend
│   ├── app.py                       # Main FastAPI server
│   ├── process_video.py             # Video processing pipeline
│   ├── __init__.py                  # Python package marker
│   ├── README.md                    # Backend documentation
│   ├── uploads/                     # Temporary upload storage
│   └── outputs/                     # Processed video storage
│
└── 📁 noise/                        # Virtual environment
    ├── Scripts/                     # Python executables
    ├── Lib/site-packages/           # Installed packages
    └── third_party/
        └── ffmpeg/                  # FFmpeg binaries
            └── ffmpeg-8.0-essentials_build/
                └── bin/
                    └── ffmpeg.exe   # Video processing tool
```

---

## 🔬 How It Works

### **Processing Pipeline (7 Steps)**

```mermaid
graph LR
    A[Upload Video] --> B[Extract Audio]
    B --> C[DeepFilterNet AI]
    C --> D[Facebook Denoiser*]
    D --> E[Speech Enhancement]
    E --> F[Silence Detection]
    F --> G[Trim & Sync]
    G --> H[Output Clean Video]
```

### **Detailed Process**

#### **Step 1: Audio Extraction** 🎧
- Extracts audio track from video
- Converts to 16kHz mono WAV (reduces memory usage)
- Uses FFmpeg for reliable extraction

#### **Step 2: DeepFilterNet Denoising** 🤖
- **Model**: DeepFilterNet3 (state-of-the-art neural network)
- **Purpose**: Removes stationary noise (fan hum, AC, static)
- **Method**: Trained on thousands of noise samples
- **Output**: Significantly cleaner audio

#### **Step 3: Facebook Denoiser** 🧠 *(Optional)*
- **Model**: DNS64 (Facebook Research)
- **Purpose**: Removes complex non-stationary noise
- **Method**: Deep learning-based real-time denoising
- **Requirement**: ~7GB RAM
- **Quality**: Superior results when enabled

#### **Step 4: Speech Enhancement** 🔊
- **Energy-based gating**: Reduces background between words
- **Band-pass EQ**: Boosts speech frequencies (1-3kHz)
- **Normalization**: Balances audio levels
- **Result**: Crystal clear voice quality

#### **Step 5: Silence Detection** ✂️
- **Algorithm**: FFmpeg silencedetect filter
- **Threshold**: -35dB, 1 second minimum
- **Output**: Timestamps of all silent segments

#### **Step 6: Video Trimming** 🎬
- Cuts video at silence boundaries
- Trims corresponding audio segments
- Maintains frame-perfect sync
- Concatenates remaining segments

#### **Step 7: Final Merge** 🔗
- Combines cleaned audio with trimmed video
- Uses codec copy (no re-encoding = faster)
- Outputs final MP4 file
- Cleans up all temporary files

---

## ⚙️ Configuration Options

### **Processing Modes**

| Option | CPU Mode | GPU Mode | GPU + FB Denoiser |
|--------|----------|----------|-------------------|
| **Speed** | ⭐⭐ Slow | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐ Fast |
| **RAM Usage** | 4-8GB | 4-8GB | 12-16GB |
| **Quality** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Requirements** | Any CPU | NVIDIA GPU | NVIDIA GPU + 16GB RAM |

### **When to Use Each Mode**

#### **CPU Mode (Default)** ✅
- ✅ No GPU available
- ✅ Limited RAM (8GB)
- ✅ Large videos (>100MB)
- ✅ Most reliable option

#### **GPU Mode** 🚀
- ✅ NVIDIA GPU with CUDA
- ✅ Faster processing (3-5x speed)
- ✅ Same quality as CPU
- ✅ Handles any video size

#### **GPU + Facebook Denoiser** 💎
- ✅ Best possible quality
- ✅ Complex noise environments
- ✅ Professional use cases
- ⚠️ Requires 16GB+ RAM
- ⚠️ May fail on large videos

---

## 📡 API Documentation

### **Base URL**
```
http://localhost:8000
```

### **Endpoints**

#### **1. Health Check**
```http
GET /
```

**Response:**
```json
{
  "status": "ok",
  "message": "AI Video Noise Removal API is running"
}
```

---

#### **2. Upload & Process Video**
```http
POST /upload
Content-Type: multipart/form-data
```

**Request Body:**
- `file` (file): Video file to process
- `use_gpu` (boolean): Enable GPU acceleration (default: false)
- `use_facebook_denoiser` (boolean): Enable FB Denoiser (default: false)

**Response:**
```json
{
  "status": "done",
  "output_file": "/download/uuid_clean_synced.mp4",
  "original_file": "/download/uuid_original.mp4",
  "statistics": {
    "original_duration": 120.5,
    "processed_duration": 95.2,
    "silence_removed_percent": 21.0,
    "segments_removed": 8
  },
  "message": "Video processed successfully"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@video.mp4" \
  -F "use_gpu=false" \
  -F "use_facebook_denoiser=false"
```

---

#### **3. Download Video**
```http
GET /download/{filename}
```

**Parameters:**
- `filename`: Name of the file to download

**Response:** Video file (video/mp4)

---

#### **4. Delete Video** *(Optional)*
```http
DELETE /cleanup/{filename}
```

**Response:**
```json
{
  "status": "ok",
  "message": "File filename.mp4 deleted successfully"
}
```

---

## 🔧 Troubleshooting

### **Common Issues**

#### **1. "CUDA out of memory" Error**
**Problem:** GPU doesn't have enough VRAM

**Solution:**
```
✅ Uncheck "Use GPU" option
✅ Use CPU mode instead
✅ Process shorter videos
✅ Close other GPU applications
```

---

#### **2. "Not enough memory" Error**
**Problem:** System RAM exhausted (usually with Facebook Denoiser)

**Solution:**
```
✅ Uncheck "Use Facebook Denoiser"
✅ Close other applications
✅ Process shorter videos (split long videos)
✅ Restart your computer to free RAM
```

---

#### **3. "ffmpeg not found" Error**
**Problem:** FFmpeg executable not located

**Solution:**
```bash
# Verify FFmpeg path in process_video.py
# Should be: noise/third_party/ffmpeg/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe

# Or install system-wide FFmpeg
# Windows: choco install ffmpeg
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg
```

---

#### **4. Server Won't Start**
**Problem:** Port 8000 already in use

**Solution:**
```bash
# Option 1: Kill process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>

# Option 2: Change port in app.py
# Line: uvicorn.run("app:app", host="0.0.0.0", port=8001)
```

---

#### **5. Video Upload Fails**
**Problem:** File too large or wrong format

**Solution:**
```
✅ Check file size (keep under 500MB for best results)
✅ Supported formats: .mp4, .avi, .mov, .mkv
✅ Check browser console for errors (F12)
✅ Try different video codec/format
```

---

#### **6. Models Not Downloading**
**Problem:** DeepFilterNet/Denoiser models fail to download

**Solution:**
```bash
# Manually download models
python -c "from df import init_df; init_df()"
python -c "from denoiser import pretrained; pretrained.dns64()"

# Check internet connection
# Check firewall settings
# Try using VPN if blocked
```

---

## 🚀 Performance Tips

### **For Faster Processing**

1. **Enable GPU** if available (3-5x faster)
2. **Keep videos under 5 minutes** for optimal speed
3. **Use 720p videos** instead of 4K (faster, same audio quality)
4. **Close other applications** to free resources
5. **Use SSD storage** for temp files

### **For Better Quality**

1. **Enable Facebook Denoiser** (if you have 16GB+ RAM)
2. **Use higher quality source videos**
3. **Ensure good microphone in original recording**
4. **Avoid heavily compressed videos**

### **For Large Videos**

1. **Split video into chunks** (use free video splitter)
2. **Process each chunk separately**
3. **Use CPU mode** (more memory efficient)
4. **Don't use Facebook Denoiser**

---

## 📊 Technical Specifications

### **AI Models Used**

| Model | Version | Size | Purpose | Training Data |
|-------|---------|------|---------|---------------|
| DeepFilterNet3 | 0.5.6 | ~5MB | Noise suppression | 10k+ hours speech |
| Facebook DNS64 | Latest | ~50MB | Advanced denoising | Real-world noise |

### **Audio Processing**

- **Sample Rate**: 16kHz (memory efficient) / 48kHz (original)
- **Channels**: Mono (converted from stereo)
- **Bit Depth**: 16-bit PCM
- **Format**: WAV (intermediate), MP4 (final)

### **Video Processing**

- **Codec**: Copy mode (no re-encoding)
- **Container**: MP4
- **Sync Method**: Frame-accurate timestamp matching
- **Tool**: FFmpeg 8.0

---

## 🎓 Credits & Acknowledgments

### **AI Models**
- **DeepFilterNet** - [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **Facebook Denoiser** - [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser)

### **Libraries**
- **FastAPI** - Modern web framework
- **PyTorch** - Deep learning framework
- **Librosa** - Audio analysis
- **FFmpeg** - Video processing

### **Author**
- **Veeresh Amaragatti** - [@Veereshamaragatti](https://github.com/Veereshamaragatti)

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Support

### **Getting Help**

1. **Check Troubleshooting Section** above
2. **Search Existing Issues** on GitHub
3. **Create New Issue** with details:
   - Error message
   - System specs (OS, RAM, GPU)
   - Video details (size, format, duration)
   - Steps to reproduce

### **Reporting Bugs**

Include:
- Full error traceback
- `pip list` output
- Video sample (if possible)
- Screenshots

---

## 🔄 Updates & Maintenance

### **Model Updates**

Models auto-download on first use and cache locally:
- **Location**: `%LOCALAPPDATA%/DeepFilterNet/` (Windows)
- **Size**: ~50MB total
- **Updates**: Automatic when package updates

### **Keeping Project Updated**

```bash
# Update Python packages
pip install --upgrade deepfilternet denoiser fastapi torch torchaudio

# Update repository
git pull origin main
```

---

## 📞 Contact

- **Repository**: [github.com/Veereshamaragatti/NoiseRemoval](https://github.com/Veereshamaragatti/NoiseRemoval)
- **Issues**: [github.com/Veereshamaragatti/NoiseRemoval/issues](https://github.com/Veereshamaragatti/NoiseRemoval/issues)

---

## 🎉 Thank You!

Thank you for using AI Video Noise Removal! If this project helped you, please consider:
- ⭐ **Starring** the repository
- 🐛 **Reporting** bugs
- 💡 **Suggesting** features
- 📢 **Sharing** with others

---

**Made with ❤️ by Veeresh Amaragatti**
