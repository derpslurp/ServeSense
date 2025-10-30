import os
import cv2
from moviepy.editor import VideoFileClip
from typing import Dict, Any


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def is_allowed_extension(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def analyze_video_basic(video_path: str, output_dir: str) -> Dict[str, Any]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    # Basic metadata via moviepy; robust to many formats
    with VideoFileClip(video_path) as clip:
        duration = float(clip.duration) if clip.duration else None
        fps = float(clip.fps) if clip.fps else None

    # Fallback frame read via OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video for analysis")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read first frame")

    h, w = frame.shape[:2]

    # Dummy annotation: draw a central crosshair to represent a placeholder pose overlay
    center = (w // 2, h // 2)
    color = (0, 255, 0)
    cv2.line(frame, (center[0] - 40, center[1]), (center[0] + 40, center[1]), color, 2)
    cv2.line(frame, (center[0], center[1] - 40), (center[0], center[1] + 40), color, 2)

    # Save annotated preview image
    base = os.path.basename(video_path)
    name, _ = os.path.splitext(base)
    preview_path = os.path.join(output_dir, f"{name}_preview.jpg")
    cv2.imwrite(preview_path, frame)

    cap.release()

    annotations = [
        {
            "frame_index": 0,
            "notes": "Placeholder annotation",
            "preview_image": os.path.basename(preview_path),
            "image_width": w,
            "image_height": h,
        }
    ]

    return {
        "duration_sec": duration,
        "frame_count": frame_count,
        "annotations": annotations,
    }


