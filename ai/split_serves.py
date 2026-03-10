"""
Test script: split a long serving video into individual one-serve clips.
Detects each serve (pose-based: wrist peak = contact) and saves clips to ./serves/

Usage:
  python split_serves.py path/to/video.mp4
  python split_serves.py path/to/video.mp4 --pre 2.5 --post 1.0

Requires: opencv-python, numpy. For best results, run from repo root or backend
so YOLO pose model is available (ultralytics + yolov8n-pose.pt); otherwise
falls back to motion-based segmentation.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# Clip window around each detected serve (seconds)
DEFAULT_PRE_SEC = 2.5   # before contact (toss + prep)
DEFAULT_POST_SEC = 1.0  # after contact (follow-through)
MIN_SERVE_GAP_SEC = 1.2  # minimum time between two serves (ignore duplicate peaks)


def get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "total_frames": total_frames, "width": width, "height": height}


def detect_contact_frames_pose(video_path: str, meta: dict) -> list[int]:
    """Use YOLO pose to find contact frame for each serve (right wrist highest)."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return []

    # Prefer backend model path
    model_paths = [
        Path(__file__).resolve().parent.parent / "backend" / "yolov8n-pose.pt",
        Path("yolov8n-pose.pt"),
    ]
    model_path = None
    for p in model_paths:
        if p.exists():
            model_path = str(p)
            break
    if not model_path:
        return []

    print("Loading YOLOv8 pose model for serve detection...")
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = meta["fps"]
    wrist_y_by_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = len(wrist_y_by_frame)
        results = model(frame, verbose=False, conf=0.3)
        y_val = None
        if len(results[0].keypoints) > 0:
            # keypoints: 0-4 face, 5-6 shoulders, 7-8 elbows, 9-10 wrists
            kp = results[0].keypoints.data[0].cpu().numpy()
            if kp.shape[0] > 10 and kp[9][2] > 0.3:  # right_wrist conf
                y_val = float(kp[9][1])
        wrist_y_by_frame.append(y_val)
    cap.release()

    # Find contact frames = local minima of wrist y (arm highest) in frame space
    ys = np.array([y if y is not None else np.nan for y in wrist_y_by_frame])
    valid_mask = ~np.isnan(ys)
    if np.sum(valid_mask) < 10:
        return []

    y_range = np.nanmax(ys) - np.nanmin(ys)
    min_gap_frames = int(MIN_SERVE_GAP_SEC * fps)
    half = min_gap_frames // 2

    contact_frames = []
    for i in range(half, len(ys) - half):
        if np.isnan(ys[i]):
            continue
        window = ys[i - half : i + half + 1]
        if np.nanmin(window) == ys[i] and (not contact_frames or i - contact_frames[-1] >= min_gap_frames):
            contact_frames.append(i)

    return contact_frames


def detect_contact_frames_motion(video_path: str, meta: dict) -> list[int]:
    """Fallback: use frame-to-frame motion to find active (serve) segments; use segment midpoints as clip centers."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = meta["fps"]
    total = meta["total_frames"]
    prev = None
    motion = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev is not None:
            diff = cv2.absdiff(prev, gray)
            motion.append(np.mean(diff))
        else:
            motion.append(0.0)
        prev = gray
    cap.release()

    if len(motion) < int(fps * 2):
        return []

    motion = np.array(motion)
    # Smooth
    k = int(fps * 0.25) | 1
    motion_smooth = np.convolve(motion, np.ones(k) / k, mode="same")
    thresh = np.percentile(motion_smooth, 75)
    min_gap = int(MIN_SERVE_GAP_SEC * fps)
    min_len = int(1.5 * fps)  # at least 1.5 s of motion = one serve

    # Find contiguous high-motion segments
    above = motion_smooth >= thresh
    segments = []
    start = None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            if i - start >= min_len:
                mid = (start + i) // 2
                segments.append(mid)
            start = None
    if start is not None and len(above) - start >= min_len:
        mid = (start + len(above)) // 2
        segments.append(mid)

    # Enforce min gap between segment centers
    out = []
    for mid in segments:
        if not out or mid - out[-1] >= min_gap:
            out.append(mid)
    return out


def extract_clip(video_path: str, start_frame: int, end_frame: int, out_path: str, meta: dict) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        out_path,
        fourcc,
        meta["fps"],
        (meta["width"], meta["height"]),
    )
    if not writer.isOpened():
        return False

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_frames = end_frame - start_frame
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="Split a long serving video into one-serve clips.")
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        default=None,
        help="Path to the serving video (or omit to be prompted; then drag-and-drop the file into the terminal)",
    )
    parser.add_argument("--pre", type=float, default=DEFAULT_PRE_SEC, help="Seconds before contact to include (default: 2.5)")
    parser.add_argument("--post", type=float, default=DEFAULT_POST_SEC, help="Seconds after contact to include (default: 1.0)")
    parser.add_argument("--out-dir", type=str, default="serves", help="Output folder for clips (default: serves)")
    parser.add_argument("--motion-only", action="store_true", help="Use motion-based detection only (no YOLO)")
    args = parser.parse_args()

    video_path = args.video
    if not video_path or not video_path.strip():
        print("Drag and drop your video file here (or paste the path), then press Enter:")
        video_path = input().strip()
    # Strip quotes Windows may add when dragging a path
    video_path = video_path.strip('"').strip("'").strip()
    if not video_path:
        print("Error: no video path provided.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"Error: file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    meta = get_video_meta(video_path)
    fps = meta["fps"]
    total_frames = meta["total_frames"]
    pre_frames = int(args.pre * fps)
    post_frames = int(args.post * fps)

    if args.motion_only:
        contact_frames = detect_contact_frames_motion(video_path, meta)
        print(f"Motion-based detection: {len(contact_frames)} serve(s) found.")
    else:
        contact_frames = detect_contact_frames_pose(video_path, meta)
        if not contact_frames:
            print("Pose detection found no serves; falling back to motion-based detection.")
            contact_frames = detect_contact_frames_motion(video_path, meta)
        print(f"Detected {len(contact_frames)} serve(s).")

    if not contact_frames:
        print("No serves detected. Try --motion-only or ensure the video shows clear serving motion.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(video_path).stem

    for i, contact in enumerate(contact_frames):
        start_frame = max(0, contact - pre_frames)
        end_frame = min(total_frames, contact + post_frames)
        out_path = out_dir / f"{base_name}_serve_{i + 1:03d}.mp4"
        ok = extract_clip(video_path, start_frame, end_frame, str(out_path), meta)
        if ok:
            print(f"Saved: {out_path}")
        else:
            print(f"Failed to write: {out_path}", file=sys.stderr)

    print(f"Done. Clips saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
