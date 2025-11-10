#!/usr/bin/env python3
"""
Complete Video Processing Pipeline:
1. Noise Removal (DeepFilterNet + Facebook Denoiser)
2. Silence Removal
3. Audio Transcription (Whisper)
4. Subtitle Generation (VTT)
"""

import os
import subprocess
import tempfile
import re
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment, effects
from typing import List, Tuple, Dict
import whisper


class VideoProcessor:
    """Unified video processing pipeline"""
    
    def __init__(self, whisper_model_name="small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing processor on {self.device.type.upper()}...")
        
        # Load Whisper model
        print(f"📝 Loading Whisper model: {whisper_model_name}")
        self.whisper_model = whisper.load_model(whisper_model_name)
        
        # Load noise removal models
        try:
            from df import enhance, init_df
            from df.model import ModelParams
            from df.io import load_audio, save_audio
            
            self.df_enhance = enhance
            self.df_load_audio = load_audio
            self.df_save_audio = save_audio
            
            print("🎯 Loading DeepFilterNet...")
            self.df_model, self.df_state, _ = init_df()
            self.df_model = self.df_model.to(self.device)
            self.df_params = ModelParams()
            self.noise_removal_available = True
        except ImportError:
            print("⚠️  DeepFilterNet not available - skipping noise removal")
            self.noise_removal_available = False
        
        try:
            from denoiser import pretrained
            from denoiser.dsp import convert_audio
            
            print("🧠 Loading Facebook Denoiser...")
            self.fb_denoiser = pretrained.dns64().to(self.device)
            self.convert_audio = convert_audio
            self.fb_denoiser_available = True
        except ImportError:
            print("⚠️  Facebook Denoiser not available - skipping advanced denoising")
            self.fb_denoiser_available = False
    
    def extract_audio(self, video_path: str, output_path: str, sample_rate: int = 48000) -> str:
        """Extract audio from video"""
        print("🎧 Extracting audio from video...")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-vn",
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    
    def denoise_audio(self, audio_path: str, output_path: str) -> str:
        """Apply noise removal using DeepFilterNet and Facebook Denoiser"""
        if not self.noise_removal_available:
            print("⚠️  Skipping noise removal (models not available)")
            # Just copy the file
            subprocess.run(["ffmpeg", "-y", "-i", audio_path, output_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return output_path
        
        print("🔊 Removing noise with DeepFilterNet...")
        # DeepFilterNet stage
        audio_data, _ = self.df_load_audio(audio_path, sr=self.df_params.sr)
        enhanced = self.df_enhance(self.df_model, self.df_state, audio_data, pad=True)
        
        temp_stage1 = output_path.replace(".wav", "_stage1.wav")
        self.df_save_audio(temp_stage1, enhanced, sr=self.df_params.sr)
        
        # Facebook Denoiser stage (if available)
        if self.fb_denoiser_available:
            print("🧠 Applying Facebook Denoiser...")
            wav, sr = torchaudio.load(temp_stage1)
            wav = self.convert_audio(wav.to(self.device), sr, 
                                    self.fb_denoiser.sample_rate, 
                                    self.fb_denoiser.chin).unsqueeze(0)
            with torch.no_grad():
                denoised = self.fb_denoiser(wav)[0]
            
            temp_stage2 = output_path.replace(".wav", "_stage2.wav")
            torchaudio.save(temp_stage2, denoised.cpu(), self.fb_denoiser.sample_rate)
            current_audio = temp_stage2
        else:
            current_audio = temp_stage1
        
        # Apply gating and EQ
        print("🎛️  Applying speech-aware gating + EQ...")
        y, sr = librosa.load(current_audio, sr=None)
        
        # Energy-based gating
        energy = librosa.feature.rms(y=y)[0]
        mask = (energy > np.median(energy) * 0.8).astype(float)
        mask = np.repeat(mask, int(len(y)/len(mask)) + 1)[:len(y)]
        y = y * (1 - 0.3 * (1 - mask))
        
        # Band-pass EQ for speech clarity
        from scipy import signal as sps
        sos = sps.butter(2, [1000, 3000], btype="band", fs=sr, output="sos")
        y = y + 0.4 * sps.sosfilt(sos, y)
        
        # Save and normalize
        sf.write(output_path, y, sr)
        normalized = effects.normalize(AudioSegment.from_file(output_path, format="wav"))
        normalized.export(output_path, format="wav")
        
        # Cleanup temp files
        for temp_file in [temp_stage1, temp_stage2]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        return output_path
    
    def detect_silence_segments(self, audio_path: str, 
                               noise_threshold: str = "-35dB",
                               min_silence_duration: float = 1.0) -> List[Tuple[float, float]]:
        """Detect silence segments in audio"""
        print("✂️  Detecting silence segments...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as logf:
            log_path = logf.name
        
        cmd = [
            "ffmpeg", "-i", audio_path,
            "-af", f"silencedetect=n={noise_threshold}:d={min_silence_duration}",
            "-f", "null", "-"
        ]
        
        with open(log_path, "w") as log_file:
            subprocess.run(cmd, stderr=log_file, stdout=subprocess.DEVNULL)
        
        # Parse silence detection log
        with open(log_path, "r") as f:
            text = f.read()
        
        os.remove(log_path)
        
        starts = [float(x) for x in re.findall(r"silence_start: (\d+\.?\d*)", text)]
        ends = [float(x) for x in re.findall(r"silence_end: (\d+\.?\d*)", text)]
        
        # Get total duration
        duration = librosa.get_duration(filename=audio_path)
        
        # Build non-silent segments
        segments = []
        prev_end = 0.0
        
        for s, e in zip(starts, ends):
            if s - prev_end > 0.25:  # Keep segments > 0.25s
                segments.append((prev_end, s))
            prev_end = e
        
        # Add final segment if needed
        if not ends or ends[-1] < duration:
            segments.append((prev_end, duration))
        
        if not segments:
            segments = [(0, duration)]
        
        print(f"📊 Found {len(segments)} non-silent segments")
        return segments
    
    def trim_and_merge(self, video_path: str, audio_path: str, 
                      segments: List[Tuple[float, float]], 
                      output_video: str, output_audio: str) -> Tuple[str, str]:
        """Trim video and audio based on segments and merge"""
        print(f"🎬 Cutting {len(segments)} segments...")
        
        temp_dir = tempfile.mkdtemp()
        video_parts = []
        audio_parts = []
        
        # Cut segments
        for i, (start, end) in enumerate(segments):
            v_part = os.path.join(temp_dir, f"v{i}.mp4")
            a_part = os.path.join(temp_dir, f"a{i}.wav")
            
            # Cut video
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
                "-i", video_path, "-c:v", "copy", "-an", v_part
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Cut audio
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
                "-i", audio_path, "-c:a", "pcm_s16le", a_part
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            video_parts.append(v_part)
            audio_parts.append(a_part)
        
        # Create concat lists
        list_v = os.path.join(temp_dir, "list_v.txt")
        list_a = os.path.join(temp_dir, "list_a.txt")
        
        with open(list_v, "w") as f:
            for v in video_parts:
                f.write(f"file '{v}'\n")
        
        with open(list_a, "w") as f:
            for a in audio_parts:
                f.write(f"file '{a}'\n")
        
        # Merge video parts
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_v, "-c", "copy", output_video
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Merge audio parts
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_a, "-c", "copy", output_audio
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Cleanup
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass
        
        return output_video, output_audio
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio using Whisper"""
        print("📝 Transcribing audio with Whisper...")
        result = self.whisper_model.transcribe(audio_path, language=None)
        print(f"🌍 Detected language: {result.get('language', 'unknown')}")
        return result
    
    def segments_to_vtt(self, segments: List[Dict], output_path: str) -> str:
        """Convert transcription segments to VTT format"""
        def fmt_ts(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = s % 60
            return f"{h:02d}:{m:02d}:{sec:06.3f}"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            for seg in segments:
                start = seg["start"]
                end = seg["end"]
                text = seg["text"].strip()
                f.write(f"{fmt_ts(start)} --> {fmt_ts(end)}\n")
                f.write(text + "\n\n")
        
        return output_path
    
    def process_video(self, input_video: str, output_dir: str,
                     remove_noise: bool = True,
                     remove_silence: bool = True,
                     transcribe: bool = True) -> Dict:
        """
        Complete pipeline: noise removal + silence removal + transcription
        
        Returns dict with paths to processed files and transcription data
        """
        print("\n" + "="*60)
        print("🎬 STARTING COMPLETE VIDEO PROCESSING PIPELINE")
        print("="*60 + "\n")
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_video))[0]
        
        temp_dir = tempfile.mkdtemp()
        result = {
            "success": False,
            "original_video": input_video,
            "processed_video": None,
            "processed_audio": None,
            "transcription": None,
            "vtt_file": None,
            "detected_language": None
        }
        
        try:
            # Step 1: Extract audio
            raw_audio = os.path.join(temp_dir, "raw_audio.wav")
            self.extract_audio(input_video, raw_audio)
            
            # Step 2: Denoise (optional)
            if remove_noise:
                clean_audio = os.path.join(temp_dir, "clean_audio.wav")
                self.denoise_audio(raw_audio, clean_audio)
            else:
                clean_audio = raw_audio
            
            # Step 3: Detect and remove silence (optional)
            if remove_silence:
                segments = self.detect_silence_segments(clean_audio)
                
                # Trim video and audio
                trimmed_video = os.path.join(temp_dir, "trimmed_video.mp4")
                trimmed_audio = os.path.join(temp_dir, "trimmed_audio.wav")
                self.trim_and_merge(input_video, clean_audio, segments, 
                                  trimmed_video, trimmed_audio)
                
                # Final output paths
                final_video = os.path.join(output_dir, f"{base_name}_processed.mp4")
                final_audio = os.path.join(output_dir, f"{base_name}_processed.wav")
                
                # Merge final video with clean audio
                print("🔗 Creating final video with clean audio...")
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", trimmed_video,
                    "-i", trimmed_audio,
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    final_video
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # Copy audio
                subprocess.run([
                    "ffmpeg", "-y", "-i", trimmed_audio, final_audio
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
            else:
                # No silence removal - just merge clean audio with video
                final_video = os.path.join(output_dir, f"{base_name}_processed.mp4")
                final_audio = os.path.join(output_dir, f"{base_name}_processed.wav")
                
                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", input_video,
                    "-i", clean_audio,
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    final_video
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                subprocess.run([
                    "ffmpeg", "-y", "-i", clean_audio, final_audio
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            result["processed_video"] = final_video
            result["processed_audio"] = final_audio
            
            # Step 4: Transcribe (optional)
            if transcribe:
                transcription = self.transcribe_audio(final_audio)
                result["transcription"] = transcription
                result["detected_language"] = transcription.get("language", "unknown")
                
                # Generate VTT
                vtt_path = os.path.join(output_dir, f"{base_name}_transcription.vtt")
                self.segments_to_vtt(transcription.get("segments", []), vtt_path)
                result["vtt_file"] = vtt_path
            
            result["success"] = True
            
            print("\n" + "="*60)
            print("✅ PROCESSING COMPLETE!")
            print("="*60)
            print(f"📹 Processed Video: {final_video}")
            print(f"🎵 Processed Audio: {final_audio}")
            if transcribe:
                print(f"📝 Transcription: {vtt_path}")
                print(f"🌍 Language: {result['detected_language']}")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
        
        finally:
            # Cleanup temp directory
            for f in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
        
        return result


if __name__ == "__main__":
    # Test the processor
    processor = VideoProcessor(whisper_model_name="small")
    
    input_video = "test_video.mp4"
    output_dir = "processed_output"
    
    result = processor.process_video(
        input_video=input_video,
        output_dir=output_dir,
        remove_noise=True,
        remove_silence=True,
        transcribe=True
    )
    
    print("\nResult:", result)
