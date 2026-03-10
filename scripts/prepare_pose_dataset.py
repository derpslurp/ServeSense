#!/usr/bin/env python3
"""
Prepare a volleyball serve pose dataset for fine-tuning YOLOv8-pose.

Uses pseudo-labeling: runs the current YOLOv8-pose on serve images/videos,
then saves predictions in YOLO pose format for training. This adapts the
model to volleyball-specific poses (arm extension, jump, contact).

Usage:
  # From serve videos (extracts frames, generates labels):
  python scripts/prepare_pose_dataset.py --videos serves/ --out datasets/volleyball_pose

  # From a folder of images:
  python scripts/prepare_pose_dataset.py --images path/to/images/ --out datasets/volleyball_pose

  # With Roboflow dataset (after download):
  python scripts/prepare_pose_dataset.py --images datasets/roboflow_volleyball/train/images --out datasets/volleyball_pose

Requires: ultralytics, opencv-python
"""
import argparse
import os
import sys
from pathlib import Path

# COCO keypoint order (17 points)
COCO_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def extract_frames_from_videos(video_dir: str, out_img_dir: str, fps_sample: float = 2.0) -> list[str]:
    """Extract frames from videos at ~fps_sample fps. Returns list of image paths."""
    try:
        import cv2
    except ImportError:
        print("Install opencv-python: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    video_dir = Path(video_dir)
    out_img_dir = Path(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in exts]
    if not videos:
        print(f"No videos found in {video_dir}", file=sys.stderr)
        return []

    image_paths = []
    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            continue
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(vfps / fps_sample))
        frame_idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                out_path = out_img_dir / f"{vp.stem}_f{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), frame)
                image_paths.append(str(out_path))
                saved += 1
            frame_idx += 1
        cap.release()
        print(f"  {vp.name}: {saved} frames")
    return image_paths


def collect_images(img_dir: str) -> list[str]:
    """Collect image paths from directory."""
    img_dir = Path(img_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [str(p) for p in img_dir.rglob("*") if p.suffix.lower() in exts]


def run_pose_and_save_labels(
    image_paths: list[str],
    out_dir: str,
    pose_model_path: str,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    val_ratio: float = 0.15,
) -> int:
    """Run pose on images, save YOLO-format labels. Splits into train/val. Returns count of labeled images."""
    try:
        from ultralytics import YOLO
        import cv2
        import random
    except ImportError as e:
        print(f"Install dependencies: pip install ultralytics opencv-python. {e}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(pose_model_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.seed(42)
    shuffled = list(image_paths)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_paths = shuffled[:n_val]
    train_paths = shuffled[n_val:]

    labeled_count = 0
    for split, paths in [("train", train_paths), ("val", val_paths)]:
        images_dir = out_dir / "images" / split
        labels_dir = out_dir / "labels" / split
        for i, img_path in enumerate(paths):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  {split}: {i + 1}/{len(paths)}...")
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            results = model(img, verbose=False, conf=conf_threshold, imgsz=imgsz)

            stem = Path(img_path).stem
            out_img = images_dir / f"{stem}.jpg"
            import shutil
            shutil.copy2(img_path, out_img)

            lines = []
            for r in results:
                if r.keypoints is None or len(r.keypoints.data) == 0:
                    continue
                kpts = r.keypoints.data[0].cpu().numpy()  # [17, 3] x, y, conf
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                else:
                    xs = kpts[:, 0]
                    ys = kpts[:, 1]
                    vis = kpts[:, 2] > 0.25
                    if not vis.any():
                        continue
                    x1, x2 = xs[vis].min(), xs[vis].max()
                    y1, y2 = ys[vis].min(), ys[vis].max()
                    pad = 20
                    x1, x2 = max(0, x1 - pad), min(w, x2 + pad)
                    y1, y2 = max(0, y1 - pad), min(h, y2 + pad)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                parts = ["0", f"{x_center:.6f}", f"{y_center:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
                for ki in range(17):
                    px = kpts[ki, 0] / w
                    py = kpts[ki, 1] / h
                    v = 2 if kpts[ki, 2] > 0.5 else (1 if kpts[ki, 2] > 0.25 else 0)
                    parts.extend([f"{px:.6f}", f"{py:.6f}", str(v)])
                lines.append(" ".join(parts))

            if lines:
                label_path = labels_dir / f"{stem}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(lines))
                labeled_count += 1

    return labeled_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare volleyball pose dataset for fine-tuning")
    parser.add_argument("--videos", help="Directory of serve videos to extract frames from")
    parser.add_argument("--images", help="Directory of images (or path to image folder)")
    parser.add_argument("--out", default="datasets/volleyball_pose", help="Output dataset directory")
    parser.add_argument("--pose-model", default=None, help="Path to pose model (default: backend/yolov8n-pose.pt)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to sample from videos")
    parser.add_argument("--conf", type=float, default=0.25, help="Pose confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    if not args.videos and not args.images:
        parser.error("Provide --videos or --images")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_pose = project_root / "backend" / "yolov8n-pose.pt"
    pose_model = args.pose_model or (str(default_pose) if default_pose.exists() else "yolov8n-pose.pt")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    if args.videos:
        print(f"Extracting frames from {args.videos} at ~{args.fps} fps...")
        img_dir = out_dir / "raw_frames"
        image_paths = extract_frames_from_videos(args.videos, str(img_dir), args.fps)
        if not image_paths:
            print("No frames extracted.", file=sys.stderr)
            sys.exit(1)
    if args.images:
        print(f"Collecting images from {args.images}...")
        image_paths = collect_images(args.images)
        if not image_paths:
            print("No images found.", file=sys.stderr)
            sys.exit(1)

    print(f"Running pose on {len(image_paths)} images (model: {pose_model})...")
    labeled = run_pose_and_save_labels(
        image_paths, str(out_dir), pose_model,
        conf_threshold=args.conf, imgsz=args.imgsz,
    )
    # Write dataset YAML
    yaml_path = out_dir / "dataset.yaml"
    yaml_content = f"""# Volleyball serve pose dataset (pseudo-labeled for fine-tuning)
# Generated by scripts/prepare_pose_dataset.py
path: {out_dir.absolute()}
train: images/train
val: images/val

kpt_shape: [17, 3]
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

names:
  0: person

kpt_names:
  0:
    - nose
    - left_eye
    - right_eye
    - left_ear
    - right_ear
    - left_shoulder
    - right_shoulder
    - left_elbow
    - right_elbow
    - left_wrist
    - right_wrist
    - left_hip
    - right_hip
    - left_knee
    - right_knee
    - left_ankle
    - right_ankle
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Done. {labeled} images with pose labels saved to {out_dir}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"\nNext: python scripts/train_pose_model.py --data {yaml_path}")


if __name__ == "__main__":
    main()
