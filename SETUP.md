# 🚀 Quick Setup Guide

> **Get started in 5 minutes!**

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.8+ installed
- ✅ 8GB+ RAM
- ✅ 5GB free disk space
- ✅ Internet connection (for downloading models)

---

## Windows Setup (Step-by-Step)

### 1. Install Python (if not installed)
Download from: https://www.python.org/downloads/
- ✅ Check "Add Python to PATH" during installation

### 2. Open PowerShell
- Press `Win + X`
- Select "Windows PowerShell" or "Terminal"

### 3. Clone or Download Project
```powershell
# Option A: Clone with Git
git clone https://github.com/Veereshamaragatti/NoiseRemoval.git
cd NoiseRemoval

# Option B: Download ZIP, extract, then:
cd path\to\NoiseRemoval
```

### 4. Create Virtual Environment
```powershell
python -m venv noise
.\noise\Scripts\Activate.ps1
```

**If you get an error about execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\noise\Scripts\Activate.ps1
```

### 5. Install Dependencies
```powershell
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch (CPU version - works on all systems)
pip install torch torchaudio

# Install AI models
pip install deepfilternet denoiser

# Install audio processing
pip install librosa soundfile pydub scipy numpy

# Install web framework
pip install fastapi uvicorn python-multipart
```

**For NVIDIA GPU users (optional, for faster processing):**
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6. Verify Installation
```powershell
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import df; print('DeepFilterNet: OK')"
python -c "import denoiser; print('Facebook Denoiser: OK')"
```

### 7. Start the Server
```powershell
cd backend
python app.py
```

You should see:
```
🚀 Starting FastAPI server...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 8. Open Web Interface
```powershell
start ..\index.html
```

**Done! 🎉**

---

## Linux Setup (Ubuntu/Debian)

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# 3. Clone project
git clone https://github.com/Veereshamaragatti/NoiseRemoval.git
cd NoiseRemoval

# 4. Create virtual environment
python3 -m venv noise
source noise/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install torch torchaudio
pip install deepfilternet denoiser
pip install librosa soundfile pydub scipy numpy
pip install fastapi uvicorn python-multipart

# 6. Start server
cd backend
python app.py

# 7. Open browser (new terminal)
xdg-open ../index.html
```

---

## macOS Setup

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python
brew install python@3.11

# 3. Clone project
git clone https://github.com/Veereshamaragatti/NoiseRemoval.git
cd NoiseRemoval

# 4. Create virtual environment
python3 -m venv noise
source noise/bin/activate

# 5. Install dependencies
pip install --upgrade pip
pip install torch torchaudio
pip install deepfilternet denoiser
pip install librosa soundfile pydub scipy numpy
pip install fastapi uvicorn python-multipart

# 6. Start server
cd backend
python app.py

# 7. Open browser (new terminal)
open ../index.html
```

---

## Common Installation Issues

### "pip not found"
```bash
# Windows
python -m ensurepip --upgrade

# Linux/Mac
sudo apt install python3-pip  # Ubuntu/Debian
brew install python  # macOS
```

### "Python not found"
- Restart your terminal/PowerShell
- Check if Python is in PATH: `python --version`
- Reinstall Python with "Add to PATH" checked

### "Module not found" errors
```bash
# Make sure virtual environment is activated
# You should see (noise) at the start of your prompt

# Windows
.\noise\Scripts\Activate.ps1

# Linux/Mac
source noise/bin/activate

# Then retry installation
pip install <package-name>
```

### Models not downloading
- Check internet connection
- Try using a VPN if in restricted region
- Manually trigger download:
```python
python -c "from df import init_df; init_df()"
python -c "from denoiser import pretrained; pretrained.dns64()"
```

---

## First Time Usage

1. **Keep both checkboxes UNCHECKED** for first test
   - This uses CPU mode (most reliable)
   - Works on any system
   
2. **Upload a SHORT video first** (< 1 minute)
   - Tests if everything works
   - Faster processing
   
3. **Check the results**
   - Compare before/after videos
   - Check statistics
   - Verify audio quality

4. **Try GPU mode** (if you have NVIDIA GPU)
   - Check "Use GPU" box
   - Should process 3-5x faster
   
5. **Try Facebook Denoiser** (if you have 16GB+ RAM)
   - Check "Use Facebook Denoiser"
   - Best quality but uses more memory

---

## Need Help?

1. Read **Troubleshooting** section in README.md
2. Check **Issues** on GitHub
3. Create new issue with:
   - Error message
   - Your OS and Python version
   - Steps you followed

---

## Ready to Use!

Your system is now set up! Here's what to do:

1. ✅ Server running? (should see "Uvicorn running on http://0.0.0.0:8000")
2. ✅ Browser open? (index.html should be loaded)
3. ✅ Upload a test video
4. ✅ Wait for processing
5. ✅ Download your clean video!

**Enjoy! 🎉**
