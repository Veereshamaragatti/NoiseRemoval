# FastAPI Video Noise Removal Backend

## 🚀 Quick Start

### 1. Install Dependencies

First, install FastAPI and Uvicorn:

```bash
pip install fastapi uvicorn python-multipart
```

### 2. Start the Server

From the `backend/` directory:

```bash
uvicorn app:app --reload
```

Or run directly:

```bash
python app.py
```

The API will be available at: `http://localhost:8000`

### 3. Open the Frontend

Open `index.html` in your browser to use the web interface.

## 📁 File Structure

```
backend/
├── app.py              # FastAPI server with upload/download endpoints
├── process_video.py    # Video processing logic (DeepFilterNet + FFmpeg)
├── uploads/            # Temporary storage for uploaded videos
├── outputs/            # Processed videos ready for download
└── README.md          # This file
```

## 🔌 API Endpoints

### `GET /`
Health check endpoint

### `POST /upload`
Upload and process a video file
- **Body**: `multipart/form-data` with `file` field
- **Returns**: `{ "status": "done", "output_file": "/download/filename.mp4" }`

### `GET /download/{filename}`
Download a processed video file

### `DELETE /cleanup/{filename}`
Delete a processed video file (optional cleanup)

## 🧪 Testing with cURL

```bash
# Upload and process a video
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_video.mp4"

# Download the result
curl -O "http://localhost:8000/download/uuid_clean_synced.mp4"
```

## 📝 Notes

- Uploaded videos are automatically deleted after processing
- Processed videos remain in `outputs/` until manually deleted
- Large videos may take several minutes to process
- GPU acceleration is used if available (CUDA)
