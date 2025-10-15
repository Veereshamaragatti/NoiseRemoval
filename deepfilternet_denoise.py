#!/usr/bin/env python3
"""
DeepFilterNet + Facebook Denoiser + Speech-Aware Gate
======================================================
✅ Removes stationary & non-stationary noise
✅ Speech-aware gating (cleans noise during speech)
✅ Natural voice tone (no thickness)
✅ GPU acceleration with CPU fallback
"""

import os, subprocess, torch, torchaudio, numpy as np
from df import enhance, init_df
from df.model import ModelParams
from df.io import load_audio, save_audio

# ---------- Input / Output ----------
input_video = "Veeresh_internship_noise.mp4"
base_name = os.path.splitext(input_video)[0]
temp_audio = "temp_audio.wav"
stage1_audio = "stage1_clean.wav"
stage2_audio = "stage2_clean.wav"
stage2p5_audio = "stage2p5_clean.wav"
final_audio = f"{base_name}_audio_clean.wav"
output_video = f"{base_name}_clean.mp4"  # prevent overwrite

# ---------- Step 1: Extract Audio ----------
if not os.path.exists(temp_audio):
    print("🎧 Extracting audio from video...")
    subprocess.run([
        "ffmpeg", "-i", input_video, "-ar", "48000", "-ac", "1",
        "-vn", temp_audio, "-y"
    ], check=True)

# ---------- Step 2: Device Selection ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using {device.type.upper()} for processing...")

# ---------- Step 3: DeepFilterNet (Stationary Noise) ----------
print("\n🎙️ Stage 1: DeepFilterNet denoising (stationary noise)...")
model, df_state, _ = init_df()
model = model.to(device)

audio_tensor, _ = load_audio(temp_audio, sr=ModelParams().sr)
enhanced = enhance(model, df_state, audio_tensor, pad=True)
save_audio(stage1_audio, enhanced, sr=ModelParams().sr)
print("✅ Stage 1 complete.")

# ---------- Step 4: Facebook Denoiser (Non-Stationary Noise) ----------
print("\n🧠 Stage 2: Facebook Denoiser cleanup (non-stationary noise)...")
from denoiser import pretrained
from denoiser.dsp import convert_audio

denoiser_model = pretrained.dns64().to(device)
wav, sr = torchaudio.load(stage1_audio)
wav = convert_audio(wav.to(device), sr, denoiser_model.sample_rate, denoiser_model.chin).unsqueeze(0)

with torch.no_grad():
    denoised = denoiser_model(wav)[0]
torchaudio.save(stage2_audio, denoised.cpu(), denoiser_model.sample_rate)
print("✅ Stage 2 complete.")

# ---------- Step 5: Speech-Aware Post Gate ----------
print("\n🔇 Stage 2.5: Speech-aware residual noise suppression...")
import librosa, soundfile as sf

y, sr = librosa.load(stage2_audio, sr=None)
energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
threshold = np.median(energy) * 0.8
mask = (energy > threshold).astype(float)
mask = np.repeat(mask, int(len(y) / len(mask)) + 1)[:len(y)]

gate_strength = 0.3  # 0.2=soft | 0.3=balanced | 0.4=strong
y_denoised = y * (1 - gate_strength * (1 - mask))

sf.write(stage2p5_audio, y_denoised, sr)
print("✅ Stage 2.5 complete (speech-aware gating applied).")

# ---------- Step 6: Natural Voice EQ + Normalization ----------
print("\n🎧 Stage 3: Natural voice EQ enhancement...")
import scipy.signal as sps
from pydub import AudioSegment, effects

def gentle_eq(data, sr):
    """Boosts presence frequencies (1–3kHz) for clarity."""
    sos = sps.butter(2, [1000, 3000], btype='band', fs=sr, output='sos')
    boosted = sps.sosfilt(sos, data)
    return data + 0.4 * boosted

y, sr = librosa.load(stage2p5_audio, sr=None)
y = gentle_eq(y, sr)
sf.write(final_audio, y, sr)

audio = AudioSegment.from_file(final_audio, format="wav")
normalized = effects.normalize(audio)
normalized.export(final_audio, format="wav")
print("✅ Stage 3 complete (natural clarity applied).")

# ---------- Step 7: Merge Clean Audio + Original Video ----------
print("\n🎬 Stage 4: Merging cleaned audio + original video...")
subprocess.run([
    "ffmpeg", "-i", input_video, "-i", final_audio,
    "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
    "-shortest", "-y", output_video
], check=True)

# ---------- Cleanup ----------
for f in [temp_audio, stage1_audio, stage2_audio, stage2p5_audio]:
    if os.path.exists(f):
        os.remove(f)

print("🧹 Cleanup done.")
print(f"\n✅ All processing complete!\nFinal clean video saved as:\n📁 {output_video}")
