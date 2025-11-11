#!/usr/bin/env python3
"""
FastAPI backend for AI video noise removal + transcription + keyword search
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from process_video import process_video
from search_index import SearchIndexManager
from groq_api import generate_summary, answer_question, generate_assessment, evaluate_short_answer
from vtt_utils import parse_vtt, load_vtt_text

app = FastAPI(title="AI Video Processing API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent  # Main project directory
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
SUBTITLE_DIR = BASE_DIR / "subtitles"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
AUDIO_DIR = BASE_DIR / "audio"  # For TTS audio files

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, SUBTITLE_DIR, TRANSCRIPT_DIR, AUDIO_DIR]:
    os.makedirs(directory, exist_ok=True)

# Supported languages for transcription
SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "te": "Telugu",
    "ta": "Tamil",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "pa": "Punjabi",
    "or": "Odia",
    "ur": "Urdu",
}

# Mount static file directories
app.mount("/subtitles", StaticFiles(directory=str(SUBTITLE_DIR)), name="subtitles")
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

# Initialize search index manager
index_manager = SearchIndexManager(vtt_root=SUBTITLE_DIR)


@app.get("/")
async def root():
    """Redirect to advanced frontend"""
    return RedirectResponse(url="/index_advanced.html")


@app.get("/index_advanced.html")
async def serve_advanced_frontend():
    """Serve the advanced frontend HTML with all features"""
    html_path = PARENT_DIR / "index_advanced.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(html_path)


@app.get("/index_merged.html")
async def serve_merged_frontend():
    """Serve the merged frontend HTML"""
    html_path = PARENT_DIR / "index_merged.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(html_path)


@app.get("/index.html")
async def serve_original_frontend():
    """Serve the original frontend HTML"""
    html_path = PARENT_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(html_path)


@app.get("/api/status")
async def api_status():
    """API health check endpoint"""
    return {
        "status": "ok", 
        "message": "AI Video Processing API is running",
        "features": ["noise_removal", "silence_trimming", "transcription", "keyword_search"]
    }


@app.get("/api/langs")
def get_langs():
    """Get supported languages for transcription"""
    return SUPPORTED_LANGS


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    use_gpu: bool = Form(False),
    use_facebook_denoiser: bool = Form(False),
    enable_transcription: bool = Form(False),
    transcription_langs: Optional[str] = Form(None)
):
    """
    Upload a video file, process it, and return the cleaned version
    
    Args:
        file: Uploaded video file
        use_gpu: Whether to use GPU for processing (requires CUDA)
        use_facebook_denoiser: Whether to use Facebook Denoiser (higher quality but more memory)
        enable_transcription: Whether to generate transcripts
        transcription_langs: Comma-separated list of language codes (e.g., "en,hi,ta")
        
    Returns:
        JSON with status and download link
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.wav', '.mp3')):
            raise HTTPException(status_code=400, detail="Only video/audio files are allowed")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())[:8]
        file_extension = os.path.splitext(file.filename)[1] or ".mp4"
        input_filename = f"{file_id}_input{file_extension}"
        output_filename = f"{file_id}_clean_synced.mp4"
        original_filename = f"{file_id}_original{file_extension}"
        
        input_path = UPLOAD_DIR / input_filename
        output_path = OUTPUT_DIR / output_filename
        original_path = OUTPUT_DIR / original_filename
        
        # Save uploaded file
        print(f"📥 Receiving upload: {file.filename}")
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"✅ File saved: {input_path}")
        
        # Copy original to outputs for comparison
        import shutil
        shutil.copy(str(input_path), str(original_path))
        
        # Parse transcription languages
        langs = []
        if enable_transcription and transcription_langs:
            requested = [code.strip() for code in transcription_langs.split(",")]
            langs = [code for code in requested if code in SUPPORTED_LANGS]
            if not langs:
                langs = ["en"]  # Default to English
        elif enable_transcription:
            langs = ["en"]
        
        # Process the video with user-selected options
        print("🚀 Starting video processing...")
        result = process_video(
            str(input_path), 
            str(output_path),
            use_gpu=use_gpu,
            use_facebook_denoiser=use_facebook_denoiser,
            enable_transcription=enable_transcription,
            transcription_langs=langs if langs else None,
            video_id=file_id
        )
        
        # Clean up input file after processing
        try:
            os.remove(input_path)
            print(f"🧹 Removed input file: {input_path}")
        except Exception as e:
            print(f"⚠️ Could not remove input file: {e}")
        
        # Prepare response
        response_data = {
            "status": "done",
            "video_id": file_id,
            "output_file": f"/download/{output_filename}",
            "original_file": f"/download/{original_filename}",
            "statistics": {
                "original_duration": result['original_duration'],
                "processed_duration": result['processed_duration'],
                "silence_removed_percent": result['silence_removed_percent'],
                "segments_removed": result['segments_removed']
            },
            "message": "Video processed successfully"
        }
        
        # Add transcription info if enabled
        if enable_transcription and 'transcript_langs' in result:
            response_data["transcription"] = {
                "enabled": True,
                "languages": result['transcript_langs'],
                "tracks": [
                    {
                        "lang": lang,
                        "label": SUPPORTED_LANGS.get(lang, lang),
                        "vtt_url": f"/subtitles/{file_id}.{lang}.vtt",
                        "audio_url": f"/audio/{file_id}.{lang}.mp3" if lang in result.get('audio_files', {}) else None
                    }
                    for lang in result['transcript_langs']
                ]
            }
            
            # Build search indexes in background
            asyncio.create_task(index_manager.ensure_indexes_for_video(file_id))
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.get("/download/{filename}")
async def download_video(filename: str):
    """
    Download a processed video file
    
    Args:
        filename: Name of the file to download
        
    Returns:
        FileResponse with the video file
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )


@app.delete("/cleanup/{filename}")
async def cleanup_file(filename: str):
    """
    Delete a processed video file (optional cleanup endpoint)
    
    Args:
        filename: Name of the file to delete
        
    Returns:
        JSON with status
    """
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {"status": "ok", "message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.get("/api/search")
def search_keyword(
    video_id: str = Query(..., description="Video identifier"),
    q: str = Query(..., min_length=1, description="Search keyword"),
    lang: str = Query("en", description="Language code")
):
    """
    Search for a keyword in video transcripts with full context
    
    Args:
        video_id: Video identifier (file_id from upload)
        q: Search keyword
        lang: Language code for search (default: en)
        
    Returns:
        JSON with search results including timestamps and text
    """
    lang = lang.lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")
    
    try:
        # Get timestamp hits from index
        timestamp_hits = index_manager.search(video_id, lang, q)
        
        # Load VTT to get full text context
        from vtt_utils import parse_vtt
        vtt_path = SUBTITLE_DIR / f"{video_id}.{lang}.vtt"
        
        if not vtt_path.exists():
            raise HTTPException(status_code=404, detail=f"Subtitles not found for video={video_id} lang={lang}")
        
        vtt_text = vtt_path.read_text(encoding="utf-8")
        cues = parse_vtt(vtt_text)
        
        # Match timestamps to cues and return full context
        results = []
        for ts in timestamp_hits:
            # Find the cue that contains this timestamp
            for start, end, text in cues:
                if start <= ts < end:
                    results.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })
                    break
        
        return {
            "video_id": video_id,
            "lang": lang,
            "keyword": q,
            "hits": results,
            "count": len(results)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/summary/{video_id}")
async def generate_video_summary(
    video_id: str,
    lang: str = Query("en", description="Language code for subtitles")
):
    """
    Generate AI-powered video summary from subtitles
    
    Args:
        video_id: Unique video identifier
        lang: Language code (default: en)
        
    Returns:
        JSON with summary, minuteByMinute breakdown, and keyPoints
    """
    try:
        # Load subtitle file
        vtt_file = SUBTITLE_DIR / f"{video_id}.{lang}.vtt"
        if not vtt_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Subtitle file not found: {vtt_file.name}. Make sure transcription is enabled."
            )
        
        # Extract full text from VTT
        subtitle_text = load_vtt_text(str(vtt_file))
        
        if not subtitle_text.strip():
            raise HTTPException(status_code=400, detail="Subtitle file is empty")
        
        # Generate summary using Groq API
        summary_data = generate_summary(subtitle_text)
        
        return {
            "status": "success",
            "video_id": video_id,
            "language": lang,
            "data": summary_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@app.post("/api/ask")
async def ask_question(
    video_id: str = Form(...),
    question: str = Form(...),
    lang: str = Form("en")
):
    """
    Ask questions about video content
    
    Args:
        video_id: Unique video identifier
        question: User's question about the video
        lang: Language code (default: en)
        
    Returns:
        JSON with answer in markdown format
    """
    try:
        # Load subtitle file
        vtt_file = SUBTITLE_DIR / f"{video_id}.{lang}.vtt"
        if not vtt_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Subtitle file not found: {vtt_file.name}"
            )
        
        # Extract full text from VTT
        subtitle_text = load_vtt_text(str(vtt_file))
        
        if not subtitle_text.strip():
            raise HTTPException(status_code=400, detail="Subtitle file is empty")
        
        # Get answer using Groq API
        answer = answer_question(subtitle_text, question)
        
        return {
            "status": "success",
            "question": question,
            "answer": answer,
            "video_id": video_id,
            "language": lang
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.post("/api/assessment/{video_id}")
async def generate_video_assessment(
    video_id: str,
    lang: str = Query("en", description="Language code for subtitles")
):
    """
    Generate educational assessment (MCQs + short questions) from video content
    
    Args:
        video_id: Unique video identifier
        lang: Language code (default: en)
        
    Returns:
        JSON with mcqs and shortQuestions arrays
    """
    try:
        # Load subtitle file
        vtt_file = SUBTITLE_DIR / f"{video_id}.{lang}.vtt"
        if not vtt_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Subtitle file not found: {vtt_file.name}"
            )
        
        # Extract full text from VTT
        subtitle_text = load_vtt_text(str(vtt_file))
        
        if not subtitle_text.strip():
            raise HTTPException(status_code=400, detail="Subtitle file is empty")
        
        # Generate assessment using Groq API
        assessment_data = generate_assessment(subtitle_text)
        
        return {
            "status": "success",
            "video_id": video_id,
            "language": lang,
            "data": assessment_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Assessment generation failed: {str(e)}")


@app.post("/api/evaluate")
async def evaluate_answer(
    question: str = Form(...),
    model_answer: str = Form(...),
    user_answer: str = Form(...)
):
    """
    Evaluate student's short answer against model answer
    
    Args:
        question: The question text
        model_answer: Expected/model answer
        user_answer: Student's submitted answer
        
    Returns:
        JSON with score (0-2) and detailed feedback
    """
    try:
        result = evaluate_short_answer(question, model_answer, user_answer)
        
        return {
            "status": "success",
            "score": result["score"],
            "feedback": result["feedback"]
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🌐 Open frontend at: http://localhost:8001/")
    print(f"🌐 Or directly: http://localhost:8001/index_merged.html")
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
