from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import uvicorn
import os
import sys
import shutil
import uuid
import logging
from typing import List, Optional, Dict, Any

from .settings import settings
from .utils.video import is_allowed_extension, analyze_video_basic
from .utils.ai_bridge import analyze_with_ai, analyze_frames as analyze_frames_ai
from .utils.benchmark import (
    save_benchmark, save_pro_benchmark, load_all_benchmarks, get_benchmark,
    delete_benchmark, compare_with_benchmark, find_best_match, get_all_benchmarks
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    # Log which ball model will be used (so user can verify)
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.abspath(os.path.join(_here, "..", ".."))
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from ai.analyze_video_yolo import _ball_model_path
        ball_path, use_custom = _ball_model_path()
        logger.info(f"Ball detection: {ball_path} ({'custom volleyball model' if use_custom else 'COCO fallback'})")
    except Exception as e:
        logger.warning(f"Could not resolve ball model at startup: {e}")
    # Mount static serving for uploads on startup once directory exists
    # Use absolute path and allow directory listing for subdirectories
    abs_uploads = os.path.abspath(settings.UPLOAD_DIR)
    try:
        # Check if already mounted
        for route in app.routes:
            if hasattr(route, 'name') and route.name == 'uploads':
                logger.info("StaticFiles /uploads already mounted")
                return
        app.mount("/uploads", StaticFiles(directory=abs_uploads, html=False), name="uploads")
        logger.info(f"Mounted /uploads to serve from: {abs_uploads}")
    except ValueError as e:
        # Already mounted (reload scenario)
        logger.debug(f"StaticFiles mount error (likely already mounted): {e}")
    except Exception as e:
        logger.warning(f"Could not mount StaticFiles: {e}")


@app.get("/health")
def health():
    """Check backend status and which ball model will be used for analysis."""
    try:
        _here = os.path.dirname(os.path.abspath(__file__))  # backend/app
        _project_root = os.path.abspath(os.path.join(_here, "..", ".."))  # backend/app -> backend -> repo root
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        from ai.analyze_video_yolo import _ball_model_path
        ball_path, use_custom = _ball_model_path()
        return {
            "status": "ok",
            "ball_model": ball_path,
            "ball_model_type": "custom volleyball" if use_custom else "COCO sports ball (fallback)",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/upload", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    camera_angle: str = Form(...),
    serve_type: str = Form(...),
    player_height_cm: Optional[float] = Form(None),
):
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

    # Save metadata (camera angle, serve type, optional player height for scale)
    metadata_path = os.path.join(settings.UPLOAD_DIR, f"{video_id}_metadata.json")
    import json
    metadata = {
        "camera_angle": camera_angle,
        "serve_type": serve_type,
        "video_id": video_id,
    }
    if player_height_cm is not None and player_height_cm > 0:
        metadata["player_height_cm"] = float(player_height_cm)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return UploadResponse(video_id=video_id, filename=file.filename, size_bytes=total)


@app.get("/analyze/{video_id}")
def analyze(video_id: str, debug: bool = Query(False, description="Include debug info if AI fails"), keep_video: bool = Query(False, description="Keep original video file after analysis")):
    # Find the file matching this video_id prefix in uploads
    try:
        matches = [f for f in os.listdir(settings.UPLOAD_DIR) if f.startswith(f"{video_id}_")]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Uploads directory not found")

    if not matches:
        raise HTTPException(status_code=404, detail="Video not found")

    video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    video_candidates = [f for f in matches if f.lower().endswith(video_extensions)]

    if not video_candidates:
        logger.error(f"No video file found for {video_id}. Matches: {matches}")
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = os.path.join(settings.UPLOAD_DIR, sorted(video_candidates)[0])

    # Load metadata (camera angle, serve type, optional player height)
    metadata_path = os.path.join(settings.UPLOAD_DIR, f"{video_id}_metadata.json")
    camera_angle = None
    serve_type = None
    player_height_m = None
    if os.path.exists(metadata_path):
        try:
            import json
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                camera_angle = metadata.get("camera_angle")
                serve_type = metadata.get("serve_type")
                ph_cm = metadata.get("player_height_cm")
                if ph_cm is not None and float(ph_cm) > 0:
                    player_height_m = float(ph_cm) / 100.0
                logger.info(f"Loaded metadata for {video_id}: camera_angle={camera_angle}, serve_type={serve_type}, player_height_m={player_height_m}")
        except Exception as e:
            logger.warning(f"Could not load metadata for {video_id}: {e}")

    # Prefer AI analysis; fall back to basic if AI fails
    try:
        result = analyze_with_ai(video_path, player_height_m=player_height_m)
        result["video_id"] = video_id
        result["camera_angle"] = camera_angle
        result["serve_type"] = serve_type
        
        # Fix image paths to be accessible via /uploads endpoint
        if "key_frames" in result:
            uploads_abs = os.path.abspath(settings.UPLOAD_DIR)
            for kf in result["key_frames"]:
                if "image_path" in kf:
                    abs_path = kf["image_path"]
                    # Ensure we have absolute path
                    if not os.path.isabs(abs_path):
                        abs_path = os.path.abspath(abs_path)
                    
                    # Check if file exists
                    if not os.path.exists(abs_path):
                        logger.warning(f"Image file not found: {abs_path}")
                    else:
                        logger.info(f"Image file exists: {abs_path}")
                    
                    # Convert to relative path from uploads directory
                    try:
                        # Use absolute paths for comparison
                        abs_uploads = os.path.abspath(uploads_abs)
                        abs_image = os.path.abspath(abs_path)
                        
                        logger.debug(f"Converting path: image={abs_image}, uploads_dir={abs_uploads}")
                        
                        # Check if image is inside uploads directory
                        try:
                            rel_path = os.path.relpath(abs_image, abs_uploads)
                            logger.debug(f"Relative path before cleanup: {rel_path}")
                            
                            # Normalize to forward slashes
                            rel_path = rel_path.replace('\\', '/').replace('//', '/')
                            
                            # Clean up any 'uploads' references
                            # Remove any occurrence of 'uploads/' at the start
                            while rel_path.startswith('uploads/'):
                                rel_path = rel_path[8:]
                            while rel_path.startswith('uploads'):
                                rel_path = rel_path[7:]
                            
                            # Remove leading slashes
                            rel_path = rel_path.lstrip('/')
                            
                            # Final path should be: /uploads/{relative_path}
                            final_path = f"/uploads/{rel_path}"
                            kf["image_path"] = final_path
                            logger.info(f"Path converted: {abs_path} -> {final_path}")
                            
                            # Verify the file can be accessed
                            test_file = os.path.join(abs_uploads, rel_path)
                            if os.path.exists(test_file):
                                logger.info(f"✓ File verified: {test_file}")
                            else:
                                logger.warning(f"✗ File not found at expected location: {test_file}")
                                
                        except ValueError as ve:
                            # Different drives or cannot make relative
                            logger.warning(f"Cannot make relative path: {abs_path} from {abs_uploads} ({ve})")
                            # Extract path after 'uploads'
                            abs_path_lower = abs_path.lower().replace('\\', '/')
                            uploads_lower = abs_uploads.lower().replace('\\', '/')
                            
                            if uploads_lower in abs_path_lower:
                                # Extract everything after uploads directory
                                idx = abs_path_lower.find(uploads_lower)
                                after_uploads = abs_path[idx + len(uploads_lower):]
                                after_uploads = after_uploads.lstrip('/').replace('\\', '/')
                                kf["image_path"] = f"/uploads/{after_uploads}"
                                logger.info(f"Fallback path (after uploads): {kf['image_path']}")
                            else:
                                # Last resort - try to extract from filename
                                rel_from_video = os.path.basename(os.path.dirname(video_path))
                                filename = os.path.basename(abs_path)
                                kf["image_path"] = f"/uploads/{rel_from_video}/{filename}" if rel_from_video else f"/uploads/{filename}"
                                logger.info(f"Last resort path: {kf['image_path']}")
                    except Exception as e:
                        logger.error(f"Error processing image path {abs_path}: {e}", exc_info=True)
                        kf["image_path"] = "/uploads/missing.jpg"  # Placeholder
        
        # Automatically compare against professional benchmarks
        logger.info(f"Checking for metrics in result. Has feedback: {'feedback' in result}, Has metrics: {'feedback' in result and 'metrics' in result.get('feedback', {})}")
        if "feedback" in result and "metrics" in result["feedback"]:
            user_metrics = result["feedback"]["metrics"]
            logger.info(f"User metrics found: {user_metrics}")
            if not user_metrics or len(user_metrics) == 0:
                logger.warning("AI ran but returned no metrics (pose/ball detection or scale may have failed). Ask user to re-analyze after ensuring YOLO models are available.")
            benchmark_match = find_best_match(user_metrics, serve_type=serve_type, camera_angle=camera_angle)
            result["pro_comparison"] = benchmark_match
            logger.info(f"Automatic pro comparison added: best match = {benchmark_match.get('summary', {}).get('best_match_name', 'None')}, score = {benchmark_match.get('summary', {}).get('best_match_score', 0)}")
        else:
            logger.warning(f"No metrics found in result. Result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
            if "feedback" in result:
                logger.warning(f"Feedback keys: {result['feedback'].keys() if isinstance(result['feedback'], dict) else 'not a dict'}")
        
        logger.info(f"AI analysis successful for {video_id}: {len(result.get('key_frames', []))} frames")
        logger.info(f"Response includes frames_analyzed: {result.get('frames_analyzed', 'MISSING')}")
        
        # Delete original video file after successful analysis (we only need the annotated frames/images)
        if not keep_video and os.path.exists(video_path):
            try:
                # Only delete video files, not the extracted frames/images
                video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
                if any(video_path.lower().endswith(ext) for ext in video_extensions):
                    os.remove(video_path)
                    logger.info(f"Deleted original video file to save space: {video_path}")
            except Exception as e:
                logger.warning(f"Could not delete video file {video_path}: {e}")
        
        return result
    except Exception as ai_exc:
        logger.error(f"AI analysis failed for {video_id}: {type(ai_exc).__name__}: {str(ai_exc)}", exc_info=True)
        try:
            result = analyze_video_basic(video_path, output_dir=settings.UPLOAD_DIR)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Video not found on disk")
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {type(exc).__name__}")
        if debug:
            # Attach limited debug context
            result["debug"] = {"ai_error": f"{type(ai_exc).__name__}: {str(ai_exc)}"}
        
        # Delete original video after basic analysis too (if not keeping)
        if not keep_video and os.path.exists(video_path):
            try:
                video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
                if any(video_path.lower().endswith(ext) for ext in video_extensions):
                    os.remove(video_path)
                    logger.info(f"Deleted original video file after basic analysis: {video_path}")
            except Exception as e:
                logger.warning(f"Could not delete video file {video_path}: {e}")
        
        return AnalysisResponse(
            video_id=video_id,
            duration_sec=result.get("duration_sec"),
            frame_count=result.get("frame_count"),
            annotations=result.get("annotations", []),
        )


# ========== BENCHMARK ENDPOINTS ==========

@app.post("/benchmarks")
async def create_benchmark(request: dict):
    """
    Create a benchmark from a professional serve video
    
    Body: { "name": str, "video_id": str, "metrics": Optional[dict] }
    If metrics are not provided, extracts them from the video's analysis.
    """
    try:
        name = request.get("name")
        video_id = request.get("video_id")
        metrics = request.get("metrics")
        
        if not name or not video_id:
            raise HTTPException(status_code=400, detail="name and video_id are required")
        
        # Get the analysis if metrics not provided
        if metrics is None:
            video_path = os.path.join(settings.UPLOAD_DIR, video_id)
            if not os.path.exists(video_path):
                raise HTTPException(status_code=404, detail="Video not found")
            
            analysis = analyze_with_ai(video_path, video_id, settings.UPLOAD_DIR)
            if "feedback" not in analysis or "metrics" not in analysis["feedback"]:
                raise HTTPException(status_code=400, detail="Could not extract metrics from video")
            
            metrics = analysis["feedback"]["metrics"]
        
        benchmark = save_benchmark(name, metrics, source_video=video_id)
        return benchmark
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating benchmark: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/benchmarks")
async def list_benchmarks():
    """Get all saved benchmarks including professional ones"""
    return get_all_benchmarks(include_pro=True)


@app.get("/benchmarks/{benchmark_id}")
async def get_benchmark_endpoint(benchmark_id: str):
    """Get a specific benchmark"""
    benchmark = get_benchmark(benchmark_id)
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return benchmark


# ========== ANALYZE FRAMES (create benchmark from screenshots) ==========

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


@app.post("/analyze-frames")
async def analyze_frames_endpoint(
    image_toss: UploadFile = File(...),
    image_contact: UploadFile = File(...),
    player_height_cm: Optional[float] = Form(None),
    toss_contact_time_s: Optional[float] = Form(None),
):
    """
    Analyze two frames (toss + contact) to extract benchmark metrics.
    Returns { "metrics": { toss_height_m, contact_height_m, elbow_height_m?, toss_contact_time_s? } }
    """
    if not image_toss.filename or not image_contact.filename:
        raise HTTPException(status_code=400, detail="Both image_toss and image_contact are required")
    for f in (image_toss, image_contact):
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported image type. Use: {', '.join(IMAGE_EXTENSIONS)}")

    prefix = str(uuid.uuid4())
    toss_path = os.path.join(settings.UPLOAD_DIR, f"{prefix}_toss{os.path.splitext(image_toss.filename)[1]}")
    contact_path = os.path.join(settings.UPLOAD_DIR, f"{prefix}_contact{os.path.splitext(image_contact.filename)[1]}")

    try:
        with open(toss_path, "wb") as out:
            out.write(await image_toss.read())
        with open(contact_path, "wb") as out:
            out.write(await image_contact.read())

        player_height_m = (float(player_height_cm) / 100.0) if player_height_cm and player_height_cm > 0 else None
        result = analyze_frames_ai(
            toss_path,
            contact_path,
            player_height_m=player_height_m,
            toss_contact_time_s=toss_contact_time_s,
        )
        return result
    finally:
        for p in (toss_path, contact_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


class ProBenchmarkCreate(BaseModel):
    name: str
    serve_type: str
    camera_angle: str
    metrics: Dict[str, Any]
    description: Optional[str] = None


@app.post("/pro-benchmarks")
async def create_pro_benchmark(body: ProBenchmarkCreate):
    """Add a professional benchmark (e.g. from frame analysis)."""
    if not body.name or not body.serve_type or not body.camera_angle:
        raise HTTPException(status_code=400, detail="name, serve_type, and camera_angle are required")
    if not body.metrics or not isinstance(body.metrics, dict):
        raise HTTPException(status_code=400, detail="metrics object is required")
    try:
        entry = save_pro_benchmark(
            name=body.name,
            metrics=body.metrics,
            serve_type=body.serve_type,
            camera_angle=body.camera_angle,
            description=body.description,
        )
        return entry
    except Exception as e:
        logger.error(f"Error creating pro benchmark: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


BENCHMARKS_IMAGES_DIR = "benchmarks"


@app.post("/pro-benchmarks/from-frames")
async def create_pro_benchmark_from_frames(
    image_toss: UploadFile = File(...),
    image_contact: UploadFile = File(...),
    name: str = Form(...),
    serve_type: str = Form(...),
    camera_angle: str = Form(...),
    player_height_cm: Optional[float] = Form(None),
    toss_contact_time_s: Optional[float] = Form(None),
    description: Optional[str] = Form(None),
):
    """
    Create a pro benchmark from toss + contact frame images.
    Analyzes frames, extracts metrics, saves images for side-by-side comparison.
    Returns the created benchmark with images paths.
    """
    if not image_toss.filename or not image_contact.filename:
        raise HTTPException(status_code=400, detail="Both image_toss and image_contact are required")
    for f in (image_toss, image_contact):
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported image type. Use: {', '.join(IMAGE_EXTENSIONS)}")

    from datetime import datetime
    bid = f"{serve_type}_{camera_angle}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Use same id for directory and benchmark entry so image paths match
    bench_dir = os.path.join(settings.UPLOAD_DIR, BENCHMARKS_IMAGES_DIR, bid)
    os.makedirs(bench_dir, exist_ok=True)

    toss_ext = os.path.splitext(image_toss.filename or "")[1].lower()
    contact_ext = os.path.splitext(image_contact.filename or "")[1].lower()
    toss_path = os.path.join(bench_dir, f"toss{toss_ext}")
    contact_path = os.path.join(bench_dir, f"contact{contact_ext}")

    try:
        with open(toss_path, "wb") as out:
            out.write(await image_toss.read())
        with open(contact_path, "wb") as out:
            out.write(await image_contact.read())

        player_height_m = (float(player_height_cm) / 100.0) if player_height_cm and player_height_cm > 0 else None
        result = analyze_frames_ai(
            toss_path,
            contact_path,
            player_height_m=player_height_m,
            toss_contact_time_s=toss_contact_time_s,
        )
        metrics = result.get("metrics", {})
        if not metrics:
            raise HTTPException(status_code=400, detail="Could not extract metrics from frames. Ensure player and ball are visible.")

        # Paths relative to uploads root for serving
        rel_toss = f"{BENCHMARKS_IMAGES_DIR}/{bid}/toss{toss_ext}"
        rel_contact = f"{BENCHMARKS_IMAGES_DIR}/{bid}/contact{contact_ext}"
        images = {
            "toss": f"/uploads/{rel_toss}",
            "contact": f"/uploads/{rel_contact}",
        }

        entry = save_pro_benchmark(
            name=name.strip(),
            metrics=metrics,
            serve_type=serve_type,
            camera_angle=camera_angle,
            description=description.strip() if description else None,
            source="Created from frames",
            player_height_m=player_height_m,
            images=images,
            benchmark_id=bid,
        )
        return entry
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating pro benchmark from frames: {e}", exc_info=True)
        for p in (toss_path, contact_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup")
def cleanup_old_files(days_old: int = Query(7, description="Delete files older than this many days"), dry_run: bool = Query(False, description="Show what would be deleted without actually deleting")):
    """Clean up old video files from uploads directory. Keeps analysis images/frames."""
    import time
    
    if not os.path.exists(settings.UPLOAD_DIR):
        return {"message": "Uploads directory not found", "deleted": []}
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    deleted = []
    errors = []
    total_size = 0
    
    try:
        for filename in os.listdir(settings.UPLOAD_DIR):
            file_path = os.path.join(settings.UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    # Only delete video files, keep images and other analysis outputs
                    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
                    if file_mtime < cutoff_time and any(filename.lower().endswith(ext) for ext in video_extensions):
                        file_size = os.path.getsize(file_path)
                        if not dry_run:
                            os.remove(file_path)
                        deleted.append({
                            "filename": filename,
                            "size_mb": round(file_size / (1024 * 1024), 2),
                            "age_days": round((time.time() - file_mtime) / (24 * 60 * 60), 1)
                        })
                        total_size += file_size
            except Exception as e:
                errors.append({"filename": filename, "error": str(e)})
        
        return {
            "message": "Dry run - no files deleted" if dry_run else f"Cleaned up {len(deleted)} video files",
            "deleted_count": len(deleted),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "deleted": deleted,
            "errors": errors if errors else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.delete("/benchmarks/{benchmark_id}")
async def delete_benchmark_endpoint(benchmark_id: str):
    """Delete a benchmark"""
    success = delete_benchmark(benchmark_id)
    if not success:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return {"message": "Benchmark deleted"}


class CompareRequest(BaseModel):
    metrics: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(extra="ignore")

@app.post("/compare/{video_id}/benchmark/{benchmark_id}")
async def compare_video_with_benchmark(
    video_id: str, 
    benchmark_id: str, 
    request: CompareRequest
):
    """
    Compare a user's video analysis against a benchmark
    """
    try:
        # Get benchmark (searches both professional and user-saved benchmarks)
        benchmark = get_benchmark(benchmark_id)
        if not benchmark:
            logger.warning(f"Benchmark not found: {benchmark_id}")
            # Try to list available benchmarks for debugging
            all_benchmarks = get_all_benchmarks()
            logger.info(f"Available benchmarks: {[b.get('id') for b in all_benchmarks]}")
            raise HTTPException(status_code=404, detail=f"Benchmark not found: {benchmark_id}")
        
        # Get user metrics - prefer from request body, fallback to video analysis
        user_metrics = None
        analysis = None
        
        # First, try to get metrics from request body (frontend passes existing metrics)
        if request and hasattr(request, 'metrics') and request.metrics:
            user_metrics = request.metrics
            logger.info(f"Using metrics from request body. Metrics keys: {list(user_metrics.keys()) if user_metrics else 'None'}")
        else:
            logger.info(f"Request metrics not available. Request: {request}, Has metrics attr: {hasattr(request, 'metrics') if request else 'No request'}")
            # Fallback: try to analyze video if it still exists
            try:
                matches = [f for f in os.listdir(settings.UPLOAD_DIR) if f.startswith(f"{video_id}_")]
                if matches:
                    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
                    for match in matches:
                        if any(match.lower().endswith(ext) for ext in video_extensions):
                            video_path = os.path.join(settings.UPLOAD_DIR, match)
                            if os.path.exists(video_path):
                                try:
                                    analysis = analyze_with_ai(video_path)
                                    if "feedback" in analysis and "metrics" in analysis.get("feedback", {}):
                                        user_metrics = analysis["feedback"]["metrics"]
                                        logger.info("Analyzed existing video file")
                                        break
                                except Exception as e:
                                    logger.warning(f"Could not analyze video: {e}")
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Error getting video: {e}")
        
        # If we still don't have metrics, raise error
        if not user_metrics:
            raise HTTPException(
                status_code=400, 
                detail="Video has been deleted after analysis. Please use the automatic comparison shown on the results page, or re-upload the video to compare with a different benchmark."
            )
        benchmark_metrics = benchmark["metrics"]
        
        comparison = compare_with_benchmark(user_metrics, benchmark_metrics)
        
        return {
            "video_id": video_id,
            "benchmark": benchmark,
            "comparison": comparison,
            "analysis": analysis  # May be None if using metrics from request body
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing video with benchmark: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Watch project root (ai/) and backend so edits to ai/*.py trigger reload without restart
    _backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _project_root = os.path.dirname(_backend_dir)
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[_project_root, _backend_dir],
    )


