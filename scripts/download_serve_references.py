#!/usr/bin/env python3
"""
Download volleyball serve images from Roboflow (CC BY 4.0) and extract toss/contact frames
for pro benchmark side-by-side comparison.

Requires: pip install roboflow ultralytics opencv-python
Set ROBOFLOW_API_KEY in the environment.

Usage:
  $env:ROBOFLOW_API_KEY = "your_key"
  python scripts/download_serve_references.py

Output: backend/uploads/benchmarks/serve_references/contact.jpg, toss.jpg
Then update literature_benchmarks.json to use these paths.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "backend" / "uploads" / "benchmarks" / "serve_references"


def download_roboflow_dataset(workspace: str, project: str, version: int, out_dir: Path) -> Path:
    """Download dataset from Roboflow. Returns path to downloaded data."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Set ROBOFLOW_API_KEY. Get it from https://app.roboflow.com/settings/api", file=sys.stderr)
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    p = rf.workspace(workspace).project(project)
    v = p.version(version)
    print(f"Downloading {project} v{version}...")
    v.download("yolov8", location=str(out_dir))
    return out_dir


def find_serve_images(dataset_dir: Path) -> list[Path]:
    """Find images that have serve-related classes in the dataset."""
    # YOLO format: data.yaml has 'names' or 'nc', folders: train/images, train/labels
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"
    if not images_dir.exists():
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
    if not images_dir.exists():
        # Try to find any images folder
        for d in dataset_dir.rglob("images"):
            if d.is_dir():
                images_dir = d
                labels_dir = d.parent / "labels"
                break

    if not images_dir.exists():
        print(f"No images folder found in {dataset_dir}", file=sys.stderr)
        return []

    # Parse class names from data.yaml
    class_names = []
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        import yaml
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
            names = data.get("names") or data.get("nc")
            if isinstance(names, dict):
                class_names = list(names.values())
            elif isinstance(names, list):
                class_names = names

    # Serve-related class indices (case-insensitive)
    serve_keywords = ["serve", "serving", "a_serve", "b_serve"]
    serve_indices = set()
    for i, name in enumerate(class_names):
        if any(kw in str(name).lower() for kw in serve_keywords):
            serve_indices.add(i)

    # If we couldn't parse, assume all images are serve-related
    if not serve_indices and class_names:
        serve_indices = set(range(len(class_names)))

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img_path in images_dir.glob(ext):
            label_path = (labels_dir / img_path.stem).with_suffix(".txt")
            if not label_path.exists():
                image_paths.append(img_path)
                continue
            try:
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            if not serve_indices or cls in serve_indices:
                                image_paths.append(img_path)
                                break
            except (ValueError, FileNotFoundError):
                image_paths.append(img_path)

    return image_paths[:50]  # Limit to 50 for speed


def score_image_toss_or_contact(image_path: Path) -> tuple[float, float]:
    """
    Run pose + ball on image. Return (toss_score, contact_score).
    Higher toss_score = ball high, arm back (toss phase).
    Higher contact_score = wrist high, arm extended (contact phase).
    """
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError:
        return 0.0, 0.0

    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0, 0.0
    h, w = img.shape[:2]

    # Pose
    pose_pt = REPO_ROOT / "backend" / "yolov8n-pose.pt"
    pose_model = YOLO(str(pose_pt)) if pose_pt.exists() else YOLO("yolov8n-pose.pt")
    pose_results = pose_model(img, verbose=False)
    if len(pose_results[0].keypoints) == 0:
        return 0.0, 0.0

    kp = pose_results[0].keypoints[0]
    if kp.conf is None or kp.conf.min() < 0.3:
        return 0.0, 0.0

    # Get wrist and head y (lower y = higher in image)
    xy = kp.xy[0].cpu().numpy()
    conf = kp.conf[0].cpu().numpy()
    # COCO keypoints: 9=ear, 10=ear, 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist
    # YOLO pose: 0-4 face, 5-6 shoulders, 7-8 elbows, 9-10 wrists
    wrist_idx = 10  # right wrist
    if conf[wrist_idx] < 0.3:
        wrist_idx = 9
    head_idx = 0
    wrist_y = xy[wrist_idx][1] if conf[wrist_idx] > 0.3 else h
    head_y = xy[head_idx][1] if conf[head_idx] > 0.3 else h * 0.5

    # Ball
    ball_pt = REPO_ROOT / "backend" / "ball_best.pt"
    use_custom = ball_pt.exists()
    ball_model = YOLO(str(ball_pt)) if use_custom else YOLO("yolov8n.pt")
    ball_classes = None if use_custom else [32]
    ball_results = ball_model(img, verbose=False, conf=0.15, classes=ball_classes)
    ball_y = h * 0.5
    if len(ball_results[0].boxes) > 0:
        box = ball_results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        ball_y = (y1 + y2) / 2.0

    # Toss: ball high (low y), wrist lower (arm back)
    toss_score = (h - ball_y) / h + (wrist_y - head_y) / h if wrist_y > head_y else (h - ball_y) / h
    # Contact: wrist high (low y, arm extended)
    contact_score = (h - wrist_y) / h

    return float(toss_score), float(contact_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download serve reference images from Roboflow")
    parser.add_argument("--workspace", default="actions-players", help="Roboflow workspace")
    parser.add_argument("--project", default="volleyball-actions", help="Project name")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--dataset-dir", default=None, help="Use existing dataset dir instead of downloading")
    args = parser.parse_args()

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists():
            print(f"Dataset dir not found: {dataset_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        download_dir = REPO_ROOT / "datasets" / "roboflow_serve_refs"
        download_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = download_roboflow_dataset(args.workspace, args.project, args.version, download_dir)
        # Roboflow creates a subfolder like "volleyball-actions-1"
        for sub in dataset_dir.iterdir():
            if sub.is_dir() and (sub / "data.yaml").exists():
                dataset_dir = sub
                break

    image_paths = find_serve_images(dataset_dir)
    if not image_paths:
        print("No serve images found. Trying all images...")
        for d in [dataset_dir / "train" / "images", dataset_dir / "images", dataset_dir]:
            if d.exists():
                image_paths = list(d.glob("*.jpg"))[:30] + list(d.glob("*.png"))[:30]
                if image_paths:
                    break

    if not image_paths:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(image_paths)} images for toss/contact...")
    scored = []
    for p in image_paths:
        try:
            t, c = score_image_toss_or_contact(p)
            scored.append((p, t, c))
        except Exception as e:
            print(f"  Skip {p.name}: {e}")

    if not scored:
        print("No images could be scored.", file=sys.stderr)
        sys.exit(1)

    best_toss = max(scored, key=lambda x: x[1])
    best_contact = max(scored, key=lambda x: x[2])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    toss_dest = OUT_DIR / "toss.jpg"
    contact_dest = OUT_DIR / "contact.jpg"
    shutil.copy(best_toss[0], toss_dest)
    shutil.copy(best_contact[0], contact_dest)
    print(f"Saved: {toss_dest}")
    print(f"Saved: {contact_dest}")

    rel_toss = "benchmarks/serve_references/toss.jpg"
    rel_contact = "benchmarks/serve_references/contact.jpg"
    images_entry = {"toss": f"/uploads/{rel_toss}", "contact": f"/uploads/{rel_contact}"}

    # Update literature_benchmarks.json
    lit_path = REPO_ROOT / "backend" / "app" / "data" / "literature_benchmarks.json"
    if lit_path.exists():
        import json
        with open(lit_path) as f:
            benchmarks = json.load(f)
        for b in benchmarks:
            b["images"] = images_entry
            if "citation" in b and "Roboflow" not in b["citation"]:
                b["citation"] = (b["citation"].rstrip(". ") + ". Images: Roboflow Volleyball Actions (CC BY 4.0).").strip()
        with open(lit_path, "w") as f:
            json.dump(benchmarks, f, indent=2)
        print(f"\nUpdated {lit_path} with serve reference images.")

    # Update pro_benchmarks.json
    pro_path = REPO_ROOT / "backend" / "app" / "data" / "pro_benchmarks.json"
    if pro_path.exists():
        import json
        with open(pro_path) as f:
            benchmarks = json.load(f)
        for b in benchmarks:
            b["images"] = images_entry
        with open(pro_path, "w") as f:
            json.dump(benchmarks, f, indent=2)
        print(f"Updated {pro_path} with serve reference images.")

    print("\nAttribution: Images from Roboflow Volleyball Actions dataset (CC BY 4.0) - actions-players/volleyball-actions")


if __name__ == "__main__":
    main()
