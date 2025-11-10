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

from process_video import process_video
from search_index import SearchIndexManager

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

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, SUBTITLE_DIR, TRANSCRIPT_DIR]:
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

# Initialize search index manager
index_manager = SearchIndexManager(vtt_root=SUBTITLE_DIR)


@app.get("/")
async def root():
    """Redirect to merged frontend"""
    return RedirectResponse(url="/index_merged.html")


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
                        "url": f"/subtitles/{file_id}.{lang}.vtt"
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
    Search for a keyword in video transcripts
    
    Args:
        video_id: Video identifier (file_id from upload)
        q: Search keyword
        lang: Language code for search (default: en)
        
    Returns:
        JSON with search results including timestamps
    """
    lang = lang.lower()
    if lang not in SUPPORTED_LANGS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")
    
    try:
        hits = index_manager.search(video_id, lang, q)
        return {
            "video_id": video_id,
            "lang": lang,
            "keyword": q,
            "hits": hits,
            "count": len(hits)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🌐 Open frontend at: http://localhost:8000/")
    print(f"🌐 Or directly: http://localhost:8000/index_merged.html")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
