# 🚀 Quick Setup Guide

> **Get started in 5 minutes!**

---

## 📋 Prerequisites Check

Before starting, ensure you have:
- ✅ **Python 3.8+** installed ([Download](https://www.python.org/downloads/))
- ✅ **8GB+ RAM** (16GB recommended for Facebook Denoiser)
- ✅ **5GB+ free disk space**
- ✅ **Internet connection** (for downloading AI models ~50MB)
- ✅ **NVIDIA GPU** (optional, for GPU acceleration)

---

---

## 🪟 Windows Setup (Recommended)

### Step 1: Install Python

1. Download Python from: https://www.python.org/downloads/
2. Run the installer
3. ✅ **IMPORTANT**: Check **"Add Python to PATH"** during installation
4. Verify installation:
   ```powershell
   python --version
   ```
   Should show: `Python 3.x.x`

### Step 2: Get the Project

**Option A: Download ZIP** (Easiest)
1. Download the project as ZIP
2. Extract to a folder (e.g., `C:\NoiseRemoval`)
3. Open PowerShell and navigate:
   ```powershell
   cd C:\NoiseRemoval
   ```

**Option B: Clone with Git**
```powershell
git clone https://github.com/Veereshamaragatti/NoiseRemoval.git
cd NoiseRemoval
```

### Step 3: Activate Virtual Environment

The `noise` folder is already a pre-configured virtual environment.

```powershell
# Activate the environment
.\noise\Scripts\Activate.ps1
```

**If you get an execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\noise\Scripts\Activate.ps1
```

You should see `(noise)` prefix in your terminal:
```
(noise) PS C:\NoiseRemoval>
```

### Step 4: Verify Installation

Check if all dependencies are installed:

```powershell
# Check Python packages
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import df; print('✅ DeepFilterNet installed')"
python -c "import denoiser; print('✅ Facebook Denoiser installed')"
python -c "import fastapi; print('✅ FastAPI installed')"
python -c "import librosa; print('✅ Librosa installed')"
```

**If any package is missing, install it:**
```powershell
pip install -r requirements.txt
```

### Step 5: Verify FFmpeg

FFmpeg is included in `noise/third_party/ffmpeg/`. Verify it works:

```powershell
.\noise\third_party\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe -version
```

Should show FFmpeg version info.

### Step 6: Start the Backend Server

```powershell
cd backend
python app.py
```

You should see:
```
🚀 Starting FastAPI server...
📁 Upload directory: ...
📁 Output directory: ...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**✅ Server is running!** Keep this terminal open.

### Step 7: Open the Web Interface

Open a **new** PowerShell window:

```powershell
cd C:\NoiseRemoval
start index.html
```

Or simply double-click `index.html` in Windows Explorer.

### Step 8: Test It Out!

1. Click **"Choose Video File"** in the browser
2. Select a video with background noise
3. (Optional) Enable GPU or Facebook Denoiser
4. Click **"Process Video"**
5. Wait for processing (may take a few minutes)
6. Compare before/after videos
7. Download the cleaned video!

**🎉 You're all set!**

---

## 🐧 Linux Setup (Ubuntu/Debian)

### Quick Installation

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python and dependencies
sudo apt install python3 python3-pip python3-venv ffmpeg -y

# 3. Navigate to project
cd NoiseRemoval

# 4. Activate virtual environment
source noise/bin/activate

# 5. Install missing packages (if any)
pip install -r requirements.txt

# 6. Start server
cd backend
python app.py

# 7. Open browser (new terminal)
xdg-open ../index.html
```

---

## 🍎 macOS Setup

```bash
# 1. Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install Python and FFmpeg
brew install python@3.11 ffmpeg

# 3. Navigate to project
cd NoiseRemoval

# 4. Activate virtual environment
source noise/bin/activate

# 5. Install missing packages (if any)
pip install -r requirements.txt

# 6. Start server
cd backend
python app.py

# 7. Open browser (new terminal)
open ../index.html
```

---

## 🚀 GPU Acceleration Setup (Optional)

For **3-5x faster processing** with NVIDIA GPU:

### Requirements
- NVIDIA GPU (GTX 1060 or better)
- CUDA 11.8 or later
- 4GB+ VRAM

### Installation

**1. Install CUDA Toolkit**
Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive

**2. Install PyTorch with CUDA**
```powershell
# Activate environment first
.\noise\Scripts\Activate.ps1

# Install GPU version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**3. Verify GPU Support**
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Should show: `CUDA available: True`

**4. Enable GPU in Web Interface**
When processing a video, check the **"Use GPU"** option.

---

## 🔧 Troubleshooting

### Issue: "Python is not recognized"
**Solution:** Add Python to PATH
1. Search "Environment Variables" in Windows
2. Edit "Path" variable
3. Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python3XX\`

### Issue: "Cannot activate virtual environment"
**Solution:** Change execution policy
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Module not found: torch"
**Solution:** Install requirements
```powershell
.\noise\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "FFmpeg not found"
**Solution:** FFmpeg is bundled in `noise/third_party/ffmpeg/`
- The code automatically uses the bundled FFmpeg
- Check if the path exists: `.\noise\third_party\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe`

### Issue: "Server won't start on port 8000"
**Solution:** Port is already in use
```powershell
# Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Or use a different port
uvicorn app:app --port 8001
```

### Issue: "Out of memory"
**Solution:** Disable Facebook Denoiser or process smaller videos
- Uncheck "Use Facebook Denoiser" in the web interface
- Facebook Denoiser needs ~7GB RAM
- Try GPU mode if available (uses VRAM instead of RAM)

### Issue: "Processing takes forever"
**Solution:** Enable GPU or reduce quality
- Enable GPU if you have NVIDIA GPU
- CPU processing can take 5-10 minutes for longer videos
- Try shorter videos first

---

## 📊 Performance Benchmarks

| Configuration | 1 min video | 5 min video | RAM Usage |
|---------------|-------------|-------------|-----------|
| CPU Only | ~2 min | ~10 min | 2-3 GB |
| CPU + Facebook | ~4 min | ~20 min | 7-8 GB |
| GPU Only | ~30 sec | ~2.5 min | 2-3 GB |
| GPU + Facebook | ~1 min | ~5 min | 4-5 GB |

*Tested on: Intel i7-10700K, NVIDIA RTX 3070, 32GB RAM*

---

## 🎯 Next Steps

After successful setup:

1. **Read the README.md** - Understand how it works
2. **Try the CLI version** - Run `python deepfilternet_denoise.py`
3. **Check backend/README.md** - Learn about the API
4. **Customize settings** - Edit `backend/process_video.py`
5. **Deploy to cloud** - Use Docker or cloud platforms

---

## 📚 Additional Resources

- [DeepFilterNet Documentation](https://github.com/Rikorose/DeepFilterNet)
- [Facebook Denoiser](https://github.com/facebookresearch/denoiser)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FFmpeg Guide](https://ffmpeg.org/documentation.html)

---

## 🆘 Still Having Issues?

1. Check if virtual environment is activated: `(noise)` should appear in terminal
2. Ensure all files are in the correct directory structure
3. Verify Python version: `python --version` (should be 3.8+)
4. Check error messages in terminal
5. Try reinstalling dependencies: `pip install -r requirements.txt --force-reinstall`

---

**Happy noise removal! 🎉**
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
