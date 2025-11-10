#!/usr/bin/env python3
"""
Video processing module - wraps the AI noise removal pipeline
"""

import os
import subprocess

# We'll set CUDA visibility dynamically in the function
import torch
import torchaudio
import numpy as np
import re
import tempfile
import librosa
import soundfile as sf
from pathlib import Path
import scipy.signal as sps
from pydub import AudioSegment, effects

# Try to import DeepFilterNet (optional)
try:
    from df import enhance, init_df
    from df.model import ModelParams
    from df.io import load_audio, save_audio
    DEEPFILTERNET_AVAILABLE = True
except ImportError:
    DEEPFILTERNET_AVAILABLE = False
    print("⚠️  DeepFilterNet not available - will skip advanced noise removal")

# Try to import Facebook Denoiser (optional)
try:
    from denoiser import pretrained
    from denoiser.dsp import convert_audio
    FACEBOOK_DENOISER_AVAILABLE = True
except ImportError:
    FACEBOOK_DENOISER_AVAILABLE = False
    print("⚠️  Facebook Denoiser not available - will skip if requested")

# Set FFmpeg path
SCRIPT_DIR = Path(__file__).parent.parent
FFMPEG_PATH = SCRIPT_DIR / "noise" / "third_party" / "ffmpeg" / "ffmpeg-8.0-essentials_build" / "bin" / "ffmpeg.exe"
FFMPEG_CMD = str(FFMPEG_PATH) if FFMPEG_PATH.exists() else "ffmpeg"

# Configure pydub to use our FFmpeg
AudioSegment.converter = FFMPEG_CMD


def process_video(
    input_video: str, 
    output_path: str, 
    use_gpu: bool = False, 
    use_facebook_denoiser: bool = False,
    enable_transcription: bool = False,
    transcription_langs: list = None,
    video_id: str = None
) -> dict:
    """
    Process video: AI noise removal + silence trimming + sync + optional transcription
    
    Args:
        input_video: Path to input video file
        output_path: Path where the cleaned video should be saved
        use_gpu: Whether to use GPU (CUDA) for processing. Default False (CPU only)
        use_facebook_denoiser: Whether to use Facebook Denoiser (requires more memory). Default False
        enable_transcription: Whether to generate transcripts and subtitles. Default False
        transcription_langs: List of language codes for transcription. Default ["en"]
        video_id: Video identifier for transcript files. Default None (auto-generated)
        
    Returns:
        dict: {
            'output_path': str,
            'original_duration': float,
            'processed_duration': float,
            'silence_removed_percent': float,
            'segments_removed': int,
            'transcription_enabled': bool,
            'transcript_langs': list (optional)
        }
    """
    print(f"🎬 Starting video processing: {input_video}")
    
    # Determine device based on user choice
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU MODE ENABLED - Using: {device_name}")
        print(f"   ✓ DeepFilterNet will use GPU")
        if use_facebook_denoiser and FACEBOOK_DENOISER_AVAILABLE:
            print(f"   ✓ Facebook Denoiser will use GPU")
        if enable_transcription:
            print(f"   ✓ Whisper transcription will use GPU")
    else:
        device = torch.device("cpu")
        if use_gpu and not torch.cuda.is_available():
            print(f"⚠️  GPU requested but CUDA not available")
        print(f"🖥️  CPU MODE - Processing will be slower")
        print(f"   ✓ All models will use CPU")
    
    print(f"⚙️  Settings:")
    print(f"   • Facebook Denoiser: {'Enabled' if use_facebook_denoiser else 'Disabled'}")
    print(f"   • Transcription: {'Enabled' if enable_transcription else 'Disabled'}")
    if enable_transcription:
        print(f"   • Languages: {', '.join(transcription_langs)}")
    print()
    
    if transcription_langs is None:
        transcription_langs = ["en"]
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # === STEP 1: Extract audio ===
        temp_audio = os.path.join(temp_dir, "input_audio.wav")
        print("🎧 Extracting audio from video...")
        # Extract shorter audio segments to reduce memory usage
        subprocess.run(
            [FFMPEG_CMD, "-i", input_video, "-ar", "16000", "-ac", "1", "-vn", temp_audio, "-y"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            check=True
        )

        # === STEP 2: Denoising ===
        print("🔊 Starting audio denoising pipeline...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear any GPU memory

        # DeepFilterNet stage (if available)
        if DEEPFILTERNET_AVAILABLE:
            print(f"   → DeepFilterNet on {device.type.upper()}...")
            model, df_state, _ = init_df()
            model = model.to(device)
            # Ensure model is actually on the correct device
            for param in model.parameters():
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
            a, _ = load_audio(temp_audio, sr=ModelParams().sr)
            enh = enhance(model, df_state, a, pad=True)
            stage1 = os.path.join(temp_dir, "stage1.wav")
            save_audio(stage1, enh, sr=ModelParams().sr)
            print(f"   ✓ DeepFilterNet complete")
        else:
            print("   ⚠️  Skipping DeepFilterNet (not installed)")
            stage1 = temp_audio

        # Facebook Denoiser (optional, uses more memory)
        if use_facebook_denoiser and FACEBOOK_DENOISER_AVAILABLE:
            print(f"   → Facebook Denoiser on {device.type.upper()}...")
            dns = pretrained.dns64().to(device)
            wav, sr = torchaudio.load(stage1)
            wav = convert_audio(wav.to(device), sr, dns.sample_rate, dns.chin).unsqueeze(0)
            with torch.no_grad():
                out = dns(wav)[0]
            stage2 = os.path.join(temp_dir, "stage2.wav")
            torchaudio.save(stage2, out.cpu() if device.type == "cuda" else out, dns.sample_rate)
            print(f"   ✓ Facebook Denoiser complete")
        else:
            if use_facebook_denoiser and not FACEBOOK_DENOISER_AVAILABLE:
                print("   ⚠️  Facebook Denoiser requested but not installed")
            else:
                print("   ⊗ Skipping Facebook Denoiser (disabled)")
            stage2 = stage1  # Use previous stage output directly

        # === STEP 3: Speech-aware gating + EQ ===
        print("🔇 Speech-aware gating + EQ...")
        y, sr = librosa.load(stage2, sr=None)
        energy = librosa.feature.rms(y=y)[0]
        mask = (energy > np.median(energy) * 0.8).astype(float)
        mask = np.repeat(mask, int(len(y)/len(mask)) + 1)[:len(y)]
        y = y * (1 - 0.3 * (1 - mask))
        sos = sps.butter(2, [1000, 3000], btype="band", fs=sr, output="sos")
        y = y + 0.4 * sps.sosfilt(sos, y)
        clean_audio = os.path.join(temp_dir, "clean_audio.wav")
        sf.write(clean_audio, y, sr)
        normalized = effects.normalize(AudioSegment.from_file(clean_audio, format="wav"))
        normalized.export(clean_audio, format="wav")

        # === STEP 4: Detect silence ===
        print("✂️ Detecting silence segments...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as logf:
            log_path = logf.name
        subprocess.run(
            [FFMPEG_CMD, "-i", clean_audio, "-af", "silencedetect=n=-35dB:d=1", "-f", "null", "-"],
            stderr=open(log_path, "w"), 
            stdout=subprocess.DEVNULL
        )
        text = open(log_path).read()
        starts = [float(x) for x in re.findall(r"silence_start: (\d+\.?\d*)", text)]
        ends = [float(x) for x in re.findall(r"silence_end: (\d+\.?\d*)", text)]
        dur = librosa.get_duration(filename=clean_audio)

        segments, prev = [], 0.0
        silence_duration = 0.0
        for s, e in zip(starts, ends):
            if s - prev > 0.25:
                segments.append((prev, s))
            else:
                silence_duration += (s - prev)  # Track skipped silences
            silence_duration += (e - s)  # Track actual silence
            prev = e
        if not ends or ends[-1] < dur:
            segments.append((prev, dur))
        if not segments:
            segments = [(0, dur)]
        
        # Calculate statistics
        original_duration = dur
        processed_duration = sum(e - s for s, e in segments)
        silence_removed_percent = (silence_duration / original_duration * 100) if original_duration > 0 else 0
        segments_removed = len(starts)
        
        print(f"📊 Statistics: Original={original_duration:.1f}s, Processed={processed_duration:.1f}s, Silence removed={silence_removed_percent:.1f}%")

        # === STEP 5: Trim video and audio per segment ===
        print(f"🎬 Cutting {len(segments)} segments and re-syncing...")
        video_parts, audio_parts = [], []
        for i, (s, e) in enumerate(segments):
            v_part = os.path.join(temp_dir, f"v{i}.mp4")
            a_part = os.path.join(temp_dir, f"a{i}.wav")
            video_parts.append(v_part)
            audio_parts.append(a_part)
            subprocess.run(
                [FFMPEG_CMD, "-ss", str(s), "-to", str(e), "-i", input_video, "-c:v", "copy", "-an", v_part, "-y"],
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            subprocess.run(
                [FFMPEG_CMD, "-ss", str(s), "-to", str(e), "-i", clean_audio, "-c:a", "pcm_s16le", a_part, "-y"],
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )

        # === STEP 6: Concatenate video and audio ===
        list_v = os.path.join(temp_dir, "list_v.txt")
        list_a = os.path.join(temp_dir, "list_a.txt")
        with open(list_v, "w") as f:
            for v in video_parts:
                f.write(f"file '{v}'\n")
        with open(list_a, "w") as f:
            for a in audio_parts:
                f.write(f"file '{a}'\n")

        merged_video = os.path.join(temp_dir, "merged_video.mp4")
        merged_audio = os.path.join(temp_dir, "merged_audio.wav")

        subprocess.run(
            [FFMPEG_CMD, "-f", "concat", "-safe", "0", "-i", list_v, "-c", "copy", merged_video, "-y"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        subprocess.run(
            [FFMPEG_CMD, "-f", "concat", "-safe", "0", "-i", list_a, "-c", "copy", merged_audio, "-y"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )

        # === STEP 7: Combine final audio + video ===
        print("🔗 Merging final clean audio and trimmed video...")
        subprocess.run([
            FFMPEG_CMD, "-i", merged_video, "-i", merged_audio,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", "-y", output_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        print(f"✅ Video processing complete: {output_path}")
        
        # === STEP 8: Optional Transcription ===
        result = {
            'output_path': output_path,
            'original_duration': round(original_duration, 2),
            'processed_duration': round(processed_duration, 2),
            'silence_removed_percent': round(silence_removed_percent, 2),
            'segments_removed': segments_removed,
            'transcription_enabled': enable_transcription
        }
        
        if enable_transcription:
            try:
                print("🎙️ Starting transcription and TTS generation...")
                from transcribe import transcribe_and_generate_audio
                
                # Generate video_id if not provided
                if video_id is None:
                    video_id = Path(output_path).stem
                
                # Set output directory
                output_dir = Path(__file__).parent
                
                # Transcribe and generate TTS audio (using same GPU setting as noise removal)
                result_files = transcribe_and_generate_audio(
                    output_path,
                    output_dir,
                    transcription_langs,
                    video_id,
                    use_gpu=use_gpu  # Pass GPU setting to Whisper
                )
                
                result['transcript_langs'] = list(result_files["subtitles"].keys())
                result['vtt_files'] = result_files["subtitles"]
                result['audio_files'] = result_files.get("audio", {})
                print(f"✅ Transcription complete for languages: {', '.join(result_files['subtitles'].keys())}")
                if result_files.get("audio"):
                    print(f"✅ TTS audio generated for: {', '.join(result_files['audio'].keys())}")
                
            except Exception as e:
                print(f"⚠️ Transcription failed: {str(e)}")
                import traceback
                traceback.print_exc()
                result['transcription_error'] = str(e)
        
        # Return statistics
        return result
        
    finally:
        # === STEP 8: Cleanup everything ===
        print("🧹 Cleaning up temporary files...")
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        if 'log_path' in locals() and os.path.exists(log_path):
            try:
                os.remove(log_path)
            except:
                pass
