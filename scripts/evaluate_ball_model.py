#!/usr/bin/env python3
"""
Evaluate ball detection on a YOLO-format volleyball dataset (e.g. from Roboflow).

Expects the dataset to have a data.yaml (train/val paths, nc, names).
Uses the same YOLOv8n model as ServeSense; if the dataset has class "ball" or one class,
metrics show how well the pretrained sports-ball (COCO 32) or a fine-tuned model does.

Usage (from repo root):
  # After downloading: python scripts/evaluate_ball_model.py
  # Or: python scripts/evaluate_ball_model.py --dataset-dir datasets/roboflow_volleyball
  # To train a small ball-only model first: python scripts/evaluate_ball_model.py --train-epochs 20
"""

import argparse
import os
import sys


def find_data_yaml(dataset_dir: str) -> str:
    """Return path to data.yaml in dataset_dir or one level down (Roboflow often uses project-name/)."""
    candidates = [
        os.path.join(dataset_dir, "data.yaml"),
        os.path.join(dataset_dir, "data.yml"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    # Single subdir (e.g. volleyball-tracker-vvvdx-1/)
    for name in os.listdir(dataset_dir) if os.path.isdir(dataset_dir) else []:
        sub = os.path.join(dataset_dir, name)
        if os.path.isdir(sub):
            for yaml_name in ("data.yaml", "data.yml"):
                p = os.path.join(sub, yaml_name)
                if os.path.isfile(p):
                    return os.path.abspath(p)
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate (and optionally train) ball detection on volleyball dataset")
    parser.add_argument("--dataset-dir", default="datasets/roboflow_volleyball", help="Dataset root (contains or contains subdir with data.yaml)")
    parser.add_argument("--model", default="yolov8n.pt", help="Model to validate (e.g. yolov8n.pt or path to fine-tuned ball model)")
    parser.add_argument("--train-epochs", type=int, default=0, help="If > 0, train a model for this many epochs then evaluate")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    data_yaml = find_data_yaml(dataset_dir)
    if not data_yaml:
        print(f"No data.yaml found in {dataset_dir}. Download a dataset first:", file=sys.stderr)
        print("  export ROBOFLOW_API_KEY=your_key", file=sys.stderr)
        print("  pip install -r scripts/requirements-datasets.txt", file=sys.stderr)
        print("  python scripts/download_roboflow_volleyball.py", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        return 1

    # Use backend model if running from repo root and default model
    if args.model == "yolov8n.pt" and not os.path.isfile(args.model):
        backend_model = os.path.join(os.path.dirname(__file__), "..", "backend", "yolov8n.pt")
        if os.path.isfile(backend_model):
            args.model = os.path.abspath(backend_model)

    if args.train_epochs > 0:
        print(f"Training for {args.train_epochs} epochs on {data_yaml}...")
        model = YOLO("yolov8n.pt")
        model.train(data=data_yaml, epochs=args.train_epochs, imgsz=640, batch=16, project="runs/ball", name="train")
        # Validate the best checkpoint
        weights = os.path.join("runs", "ball", "train", "weights", "best.pt")
        if os.path.isfile(weights):
            model = YOLO(weights)
        print("Running validation...")
    else:
        model = YOLO(args.model)

    results = model.val(data=data_yaml, imgsz=640, batch=16)
    if hasattr(results, "box") and results.box is not None:
        mAP50 = getattr(results.box, "map50", None) or getattr(results.box, "maps", 0)
        print(f"\nValidation mAP50: {mAP50}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
