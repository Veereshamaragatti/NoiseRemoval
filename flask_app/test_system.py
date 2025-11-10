#!/usr/bin/env python3
"""
Test script to verify the video processing pipeline
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    
    packages = [
        ("flask", "Flask"),
        ("torch", "PyTorch"),
        ("whisper", "OpenAI Whisper"),
        ("pydub", "Pydub"),
        ("deep_translator", "Deep Translator"),
        ("gtts", "gTTS"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
    ]
    
    optional_packages = [
        ("df", "DeepFilterNet"),
        ("denoiser", "Facebook Denoiser"),
    ]
    
    success = True
    
    # Test required packages
    for module, name in packages:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            success = False
    
    # Test optional packages
    print("\nOptional packages (for noise removal):")
    for module, name in optional_packages:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - Not installed (noise removal will be skipped)")
    
    return success

def test_ffmpeg():
    """Test if ffmpeg is available"""
    print("\nTesting FFmpeg...")
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg: {version}")
            return True
        else:
            print("❌ FFmpeg found but not working properly")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg - NOT FOUND in PATH")
        print("   Install from: https://ffmpeg.org/download.html")
        return False

def test_gpu():
    """Test if GPU/CUDA is available"""
    print("\nTesting GPU/CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return False

def test_video_processor():
    """Test if VideoProcessor can be initialized"""
    print("\nTesting VideoProcessor...")
    try:
        from video_processor import VideoProcessor
        print("✅ VideoProcessor class imported successfully")
        
        # Try to initialize (this will load models)
        print("   Attempting to initialize (this may take a minute)...")
        processor = VideoProcessor(whisper_model_name="base")  # Use smallest model for test
        print("✅ VideoProcessor initialized successfully")
        return True
    except Exception as e:
        print(f"❌ VideoProcessor initialization failed: {e}")
        return False

def main():
    print("="*60)
    print("VIDEO PROCESSING PIPELINE - SYSTEM CHECK")
    print("="*60)
    print()
    
    results = []
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test FFmpeg
    results.append(("FFmpeg", test_ffmpeg()))
    
    # Test GPU
    results.append(("GPU/CUDA", test_gpu()))
    
    # Test VideoProcessor
    # results.append(("VideoProcessor", test_video_processor()))  # Commented out as it downloads models
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_required_pass = all(result for name, result in results if name in ["Package Imports", "FFmpeg"])
    
    print()
    if all_required_pass:
        print("✅ System is ready for video processing!")
        print("\nRun the app with:")
        print("   python app.py")
        print("\nThen open: http://127.0.0.1:5000")
    else:
        print("❌ Some required components are missing.")
        print("   Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    print("="*60)
    
    return 0 if all_required_pass else 1

if __name__ == "__main__":
    sys.exit(main())
