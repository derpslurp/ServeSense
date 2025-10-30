from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import shutil
import uuid
from typing import List, Optional

from .settings import settings
from .utils.video import is_allowed_extension, analyze_video_basic


class UploadResponse(BaseModel):
    video_id: str
    filename: str
    size_bytes: int


class AnalysisResponse(BaseModel):
    video_id: str
    duration_sec: Optional[float]
    frame_count: Optional[int]
    annotations: List[dict]


app = FastAPI(title="ServeSense Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def ensure_upload_dir() -> None:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    # Mount static serving for uploads on startup once directory exists
    try:
        app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
    except Exception:
        # In reload scenarios, it may already be mounted
        pass


@app.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not is_allowed_extension(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: mp4, mov, avi, mkv")

    # Stream to disk with size limit enforcement
    video_id = str(uuid.uuid4())
    target_path = os.path.join(settings.UPLOAD_DIR, f"{video_id}_{file.filename}")

    total = 0
    try:
        with open(target_path, "wb") as out:
            while True:
                chunk = await file.read(settings.CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.MAX_FILE_SIZE_BYTES:
                    # Remove partial file
                    out.close()
                    try:
                        os.remove(target_path)
                    except OSError:
                        pass
                    raise HTTPException(status_code=413, detail="File too large")
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        # Cleanup on unexpected errors
        try:
            if os.path.exists(target_path):
                os.remove(target_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save file: {type(exc).__name__}")

    return UploadResponse(video_id=video_id, filename=file.filename, size_bytes=total)


@app.get("/analyze/{video_id}", response_model=AnalysisResponse)
def analyze(video_id: str):
    # Find the file matching this video_id prefix in uploads
    try:
        matches = [f for f in os.listdir(settings.UPLOAD_DIR) if f.startswith(f"{video_id}_")]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Uploads directory not found")

    if not matches:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = os.path.join(settings.UPLOAD_DIR, matches[0])

    try:
        result = analyze_video_basic(video_path, output_dir=settings.UPLOAD_DIR)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Video not found on disk")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {type(exc).__name__}")

    return AnalysisResponse(
        video_id=video_id,
        duration_sec=result.get("duration_sec"),
        frame_count=result.get("frame_count"),
        annotations=result.get("annotations", []),
    )


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)


