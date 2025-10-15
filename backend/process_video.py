#!/usr/bin/env python3
"""
Video processing module - wraps the AI noise removal pipeline
"""

import os
import subprocess
import torch
import torchaudio
import numpy as np
import re
import tempfile
import librosa
import soundfile as sf
from pathlib import Path
from df import enhance, init_df
from df.model import ModelParams
from df.io import load_audio, save_audio
import scipy.signal as sps
from pydub import AudioSegment, effects

# Set FFmpeg path
SCRIPT_DIR = Path(__file__).parent.parent
FFMPEG_PATH = SCRIPT_DIR / "noise" / "third_party" / "ffmpeg" / "ffmpeg-8.0-essentials_build" / "bin" / "ffmpeg.exe"
FFMPEG_CMD = str(FFMPEG_PATH) if FFMPEG_PATH.exists() else "ffmpeg"

# Configure pydub to use our FFmpeg
AudioSegment.converter = FFMPEG_CMD


def process_video(input_video: str, output_path: str) -> str:
    """
    Process video: AI noise removal + silence trimming + sync
    
    Args:
        input_video: Path to input video file
        output_path: Path where the cleaned video should be saved
        
    Returns:
        str: Path to the processed output video
    """
    print(f"🎬 Starting video processing: {input_video}")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # === STEP 1: Extract audio ===
        temp_audio = os.path.join(temp_dir, "input_audio.wav")
        print("🎧 Extracting audio from video...")
        subprocess.run(
            [FFMPEG_CMD, "-i", input_video, "-ar", "48000", "-ac", "1", "-vn", temp_audio, "-y"],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            check=True
        )

        # === STEP 2: Denoising ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using {device.type.upper()} for processing...")

        model, df_state, _ = init_df()
        model = model.to(device)
        a, _ = load_audio(temp_audio, sr=ModelParams().sr)
        enh = enhance(model, df_state, a, pad=True)
        stage1 = os.path.join(temp_dir, "stage1.wav")
        save_audio(stage1, enh, sr=ModelParams().sr)

        print("🧠 Facebook Denoiser (non-stationary)...")
        from denoiser import pretrained
        from denoiser.dsp import convert_audio
        dns = pretrained.dns64().to(device)
        wav, sr = torchaudio.load(stage1)
        wav = convert_audio(wav.to(device), sr, dns.sample_rate, dns.chin).unsqueeze(0)
        with torch.no_grad():
            out = dns(wav)[0]
        stage2 = os.path.join(temp_dir, "stage2.wav")
        torchaudio.save(stage2, out.cpu(), dns.sample_rate)

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
        for s, e in zip(starts, ends):
            if s - prev > 0.25:
                segments.append((prev, s))
            prev = e
        if not ends or ends[-1] < dur:
            segments.append((prev, dur))
        if not segments:
            segments = [(0, dur)]

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
        return output_path
        
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
