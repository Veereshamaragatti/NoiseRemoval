#!/usr/bin/env python3
"""
Minimal test to verify routes are working
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
import uvicorn

app = FastAPI()

BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent

@app.get("/")
async def root():
    return RedirectResponse(url="/test.html")

@app.get("/test.html")
async def test_page():
    return {"message": "Route is working!", "parent_dir": str(PARENT_DIR)}

@app.get("/index_merged.html")
async def serve_merged():
    html_path = PARENT_DIR / "index_merged.html"
    print(f"Trying to serve: {html_path}")
    print(f"File exists: {html_path.exists()}")
    if not html_path.exists():
        return {"error": "File not found", "path": str(html_path)}
    return FileResponse(html_path)

if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"PARENT_DIR: {PARENT_DIR}")
    print(f"index_merged.html exists: {(PARENT_DIR / 'index_merged.html').exists()}")
    uvicorn.run(app, host="127.0.0.1", port=8001)
