#!/usr/bin/env python3
"""
FastAPI backend for AI video noise removal
"""

import os
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from process_video import process_video

app = FastAPI(title="AI Video Noise Removal API")

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
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "AI Video Noise Removal API is running"}


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    use_gpu: bool = Form(False),
    use_facebook_denoiser: bool = Form(False)
):
    """
    Upload a video file, process it, and return the cleaned version
    
    Args:
        file: Uploaded video file
        use_gpu: Whether to use GPU for processing (requires CUDA)
        use_facebook_denoiser: Whether to use Facebook Denoiser (higher quality but more memory)
        
    Returns:
        JSON with status and download link
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Only video files are allowed")
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        input_filename = f"{file_id}_input{file_extension}"
        output_filename = f"{file_id}_clean_synced.mp4"
        
        input_path = UPLOAD_DIR / input_filename
        output_path = OUTPUT_DIR / output_filename
        
        # Save uploaded file
        print(f"📥 Receiving upload: {file.filename}")
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"✅ File saved: {input_path}")
        
        # Process the video with user-selected options
        print("🚀 Starting video processing...")
        result_path = process_video(
            str(input_path), 
            str(output_path),
            use_gpu=use_gpu,
            use_facebook_denoiser=use_facebook_denoiser
        )
        
        # Clean up input file after processing
        try:
            os.remove(input_path)
            print(f"🧹 Removed input file: {input_path}")
        except Exception as e:
            print(f"⚠️ Could not remove input file: {e}")
        
        return JSONResponse(content={
            "status": "done",
            "output_file": f"/download/{output_filename}",
            "message": "Video processed successfully"
        })
        
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


if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    print(f"📁 Upload directory: {UPLOAD_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
