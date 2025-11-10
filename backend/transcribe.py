#!/usr/bin/env python3
"""
Transcription module using Whisper for speech-to-text in multiple languages
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json
import importlib

# Supported Indian languages + English
SUPPORTED_LANGS = [
    "en", "hi", "kn", "te", "ta", "ml", "mr", "gu", "bn", "pa", "or", "ur"
]

# FFmpeg path configuration
SCRIPT_DIR = Path(__file__).parent.parent
FFMPEG_PATH = SCRIPT_DIR / "noise" / "third_party" / "ffmpeg" / "ffmpeg-8.0-essentials_build" / "bin" / "ffmpeg.exe"

if FFMPEG_PATH.exists():
    ffmpeg_dir = str(FFMPEG_PATH.parent)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_BINARY"] = str(FFMPEG_PATH)
    print(f"✅ FFmpeg configured at: {FFMPEG_PATH}")


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS.mmm for VTT."""
    ms = int(round((seconds - int(seconds)) * 1000))
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_vtt(segments: List[Dict[str, Any]], out_path: Path):
    """Write list of {start, end, text} segments into .vtt format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip().replace("-->", "→")
            f.write(f"{start} --> {end}\n{text}\n\n")


def translate_segments(segments: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
    """
    Translate subtitle segments into target language using Google Translate.
    
    Args:
        segments: List of subtitle segments with start, end, text
        target_lang: Target language code
        
    Returns:
        Translated segments
    """
    if target_lang == "en":
        return segments

    try:
        module = importlib.import_module("deep_translator")
        GoogleTranslator = getattr(module, "GoogleTranslator", None)
    except Exception:
        print(f"⚠️ deep-translator not available, skipping translation for {target_lang}")
        return segments

    if GoogleTranslator is None:
        print(f"⚠️ GoogleTranslator not found, skipping translation for {target_lang}")
        return segments

    print(f"🔁 Translating subtitles to {target_lang}...")
    translator = GoogleTranslator(source='auto', target=target_lang)
    out = []
    delim = " ||| "
    joined = delim.join([s["text"] for s in segments])

    try:
        translated = translator.translate(joined)
        parts = [p.strip() for p in translated.split("|||")]
        if len(parts) != len(segments):
            raise ValueError("Batch translation mismatch — using fallback")
    except Exception as e:
        print(f"⚠️ Batch translation failed for {target_lang}: {e}")
        parts = []
        for s in segments:
            try:
                parts.append(translator.translate(s["text"]))
            except Exception as ex:
                print(f"⚠️ Line translation failed ({target_lang}): {ex}")
                parts.append(s["text"])

    for s, t in zip(segments, parts):
        out.append({"start": s["start"], "end": s["end"], "text": t})

    print(f"✅ Translated {len(out)} segments to {target_lang}")
    return out


def transcribe_core(audio_path: str, use_gpu: bool = False) -> List[Dict[str, Any]]:
    """
    Transcribe a single audio/video file using Whisper.
    
    Args:
        audio_path: Path to audio/video file
        use_gpu: Whether to use GPU for transcription (default: False)
        
    Returns:
        List of segments with start, end, text
    """
    try:
        import whisper
        import torch
    except ImportError:
        raise RuntimeError("Whisper not installed. Run: pip install -U openai-whisper")

    print(f"🎙️ Transcribing: {audio_path}")
    
    # Set device based on user choice
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using GPU (CUDA) for Whisper transcription...")
    else:
        device = "cpu"
        if use_gpu and not torch.cuda.is_available():
            print(f"⚠️ GPU requested but not available. Using CPU for transcription...")
        else:
            print(f"🚀 Using CPU for Whisper transcription...")
    
    model_name = os.environ.get("WHISPER_MODEL", "small")
    model = whisper.load_model(model_name, device=device)

    # Transcribe using ffmpeg under the hood
    result = model.transcribe(audio_path, task="transcribe", fp16=(device == "cuda"))
    segments = [
        {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"]}
        for seg in result.get("segments", [])
    ]

    if not segments and result.get("text"):
        segments = [{"start": 0.0, "end": 0.01, "text": result["text"]}]

    print(f"✅ Transcription complete — {len(segments)} segments.")
    return segments


def transcribe_to_vtt_many(
    media_path: str, 
    vtt_dir: Path, 
    langs: List[str], 
    video_id: str = None,
    use_gpu: bool = False
) -> Dict[str, str]:
    """
    Transcribes and translates media into multiple languages.
    
    Args:
        media_path: Path to media file
        vtt_dir: Directory to save VTT files
        langs: List of language codes to generate
        video_id: Video identifier (optional, defaults to filename)
        use_gpu: Whether to use GPU for transcription (default: False)
        
    Returns:
        Dict mapping language code to VTT path
    """
    p = Path(media_path)
    if video_id is None:
        video_id = p.stem

    print(f"🎬 Starting transcription for: {p.name}")
    base_segments = transcribe_core(media_path, use_gpu=use_gpu)

    out = {}
    for code in langs:
        if code not in SUPPORTED_LANGS:
            print(f"⚠️ Skipping unsupported language: {code}")
            continue

        try:
            if code == "en":
                segs = base_segments
            else:
                segs = translate_segments(base_segments, code)
        except Exception as e:
            print(f"⚠️ Translation failed for {code}: {e}")
            segs = base_segments

        vtt_path = vtt_dir / f"{video_id}.{code}.vtt"
        write_vtt(segs, vtt_path)
        out[code] = str(vtt_path)
        print(f"💾 Saved {code} subtitles → {vtt_path.name}")

    # Save manifest for indexing
    manifest = {
        "video_id": video_id,
        "media_file": os.path.basename(media_path),
        "langs": list(out.keys())
    }
    with (vtt_dir / f"{video_id}.manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"📜 Manifest saved for {video_id}")
    return out


def generate_tts_audio(
    segments: List[Dict[str, Any]], 
    lang_code: str,
    output_audio_path: Path,
    original_duration_ms: int
) -> str:
    """
    Generate time-aligned TTS audio from translated segments.
    
    Args:
        segments: List of segments with start, end, text
        lang_code: Language code for TTS
        output_audio_path: Path to save the generated audio
        original_duration_ms: Original audio duration in milliseconds
        
    Returns:
        Path to generated audio file
    """
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        import uuid
        import tempfile
    except ImportError:
        print("⚠️ gTTS or pydub not installed. Skipping TTS generation.")
        return None
    
    print(f"🎵 Generating TTS audio for {lang_code}...")
    
    # Build timeline
    timeline = AudioSegment.silent(duration=0)
    cursor_ms = 0
    temp_dir = Path(tempfile.gettempdir())
    
    # Map lang codes to gTTS codes
    GTTS_LANG_MAP = {
        "en": "en", "hi": "hi", "kn": "kn", "te": "te", 
        "ta": "ta", "ml": "ml", "mr": "mr", "gu": "gu",
        "bn": "bn", "pa": "pa", "ur": "ur"
    }
    
    tts_lang = GTTS_LANG_MAP.get(lang_code, "en")
    
    for seg in sorted(segments, key=lambda s: s["start"]):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        target_dur = end_ms - start_ms
        text = seg.get("text", "").strip()
        
        # Add silence before this segment
        if start_ms > cursor_ms:
            timeline += AudioSegment.silent(duration=(start_ms - cursor_ms))
            cursor_ms = start_ms
        
        if target_dur <= 0 or not text:
            if target_dur > 0:
                timeline += AudioSegment.silent(duration=target_dur)
                cursor_ms += target_dur
            continue
        
        try:
            # Generate TTS
            seg_id = uuid.uuid4().hex[:8]
            tmp_mp3 = temp_dir / f"tts_{seg_id}.mp3"
            
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            tts.save(str(tmp_mp3))
            
            # Load audio
            tts_audio = AudioSegment.from_file(tmp_mp3, format="mp3")
            actual_dur = len(tts_audio)
            
            # Time-stretch to match target duration
            if actual_dur > 0:
                speed_ratio = actual_dur / target_dur
                
                # Use ffmpeg for time-stretching
                stretched_wav = temp_dir / f"stretched_{seg_id}.wav"
                
                # Build atempo filter chain (supports 0.5-2.0 range)
                atempo_filters = []
                while speed_ratio > 2.0:
                    atempo_filters.append("atempo=2.0")
                    speed_ratio /= 2.0
                while speed_ratio < 0.5:
                    atempo_filters.append("atempo=0.5")
                    speed_ratio /= 0.5
                atempo_filters.append(f"atempo={speed_ratio:.6f}")
                
                atempo_chain = ",".join(atempo_filters)
                
                # Export to wav first
                tmp_wav = temp_dir / f"in_{seg_id}.wav"
                tts_audio.export(tmp_wav, format="wav")
                
                # Apply time-stretch
                import subprocess
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(tmp_wav),
                    "-filter:a", atempo_chain,
                    str(stretched_wav)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Load stretched audio
                processed_audio = AudioSegment.from_file(stretched_wav, format="wav")
                
                # Fine-tune duration
                if len(processed_audio) > target_dur:
                    processed_audio = processed_audio[:target_dur]
                elif len(processed_audio) < target_dur:
                    processed_audio += AudioSegment.silent(duration=(target_dur - len(processed_audio)))
                
                timeline += processed_audio
                
                # Cleanup temp files
                for f in [tmp_mp3, tmp_wav, stretched_wav]:
                    if f.exists():
                        f.unlink()
            else:
                timeline += AudioSegment.silent(duration=target_dur)
                
            cursor_ms += target_dur
            
        except Exception as e:
            print(f"⚠️ TTS failed for segment: {e}")
            timeline += AudioSegment.silent(duration=target_dur)
            cursor_ms += target_dur
    
    # Pad or trim to match original duration
    if cursor_ms < original_duration_ms:
        timeline += AudioSegment.silent(duration=(original_duration_ms - cursor_ms))
    elif cursor_ms > original_duration_ms:
        timeline = timeline[:original_duration_ms]
    
    # Export
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    timeline.export(str(output_audio_path), format="mp3")
    print(f"✅ TTS audio saved: {output_audio_path.name}")
    
    return str(output_audio_path)


def transcribe_and_generate_audio(
    media_path: str,
    output_dir: Path,
    langs: List[str],
    video_id: str = None,
    use_gpu: bool = False
) -> Dict[str, Dict[str, str]]:
    """
    Transcribe, translate, and generate TTS audio for multiple languages.
    
    Args:
        media_path: Path to media file
        output_dir: Directory for outputs (subtitles/, audio/)
        langs: List of language codes
        video_id: Video identifier
        use_gpu: Use GPU for transcription
        
    Returns:
        Dict with 'subtitles' and 'audio' paths for each language
    """
    from pydub import AudioSegment
    
    p = Path(media_path)
    if video_id is None:
        video_id = p.stem
    
    # Create output directories
    vtt_dir = output_dir / "subtitles"
    audio_dir = output_dir / "audio"
    vtt_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original audio duration
    try:
        original_audio = AudioSegment.from_file(media_path)
        original_duration_ms = len(original_audio)
    except:
        original_duration_ms = 0
    
    # Transcribe
    print(f"🎬 Starting transcription and TTS generation for: {p.name}")
    base_segments = transcribe_core(media_path, use_gpu=use_gpu)
    
    result = {"subtitles": {}, "audio": {}}
    
    for code in langs:
        if code not in SUPPORTED_LANGS:
            print(f"⚠️ Skipping unsupported language: {code}")
            continue
        
        try:
            # Translate segments
            if code == "en":
                segs = base_segments
            else:
                segs = translate_segments(base_segments, code)
            
            # Save VTT
            vtt_path = vtt_dir / f"{video_id}.{code}.vtt"
            write_vtt(segs, vtt_path)
            result["subtitles"][code] = str(vtt_path)
            print(f"💾 Saved {code} subtitles → {vtt_path.name}")
            
            # Generate TTS audio
            audio_path = audio_dir / f"{video_id}.{code}.mp3"
            audio_file = generate_tts_audio(segs, code, audio_path, original_duration_ms)
            if audio_file:
                result["audio"][code] = audio_file
                
        except Exception as e:
            print(f"⚠️ Processing failed for {code}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save manifest
    manifest = {
        "video_id": video_id,
        "media_file": os.path.basename(media_path),
        "langs": list(result["subtitles"].keys())
    }
    with (vtt_dir / f"{video_id}.manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"📜 Manifest saved for {video_id}")
    return result
