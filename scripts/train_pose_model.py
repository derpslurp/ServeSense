#!/usr/bin/env python3
"""
Fine-tune YOLOv8-pose on volleyball serve data.

Uses the dataset prepared by prepare_pose_dataset.py. Trains from the
pretrained yolov8n-pose.pt (or yolov8s-pose.pt for better accuracy).

Usage:
  python scripts/train_pose_model.py --data datasets/volleyball_pose/dataset.yaml
  python scripts/train_pose_model.py --data datasets/volleyball_pose/dataset.yaml --model yolov8s-pose.pt --epochs 50

After training, set POSE_MODEL to the best.pt path to use the fine-tuned model:
  POSE_MODEL= runs/pose/volleyball/weights/best.pt

Requires: ultralytics
"""
import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8-pose on volleyball serve data")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml (from prepare_pose_dataset.py)")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="Base model to fine-tune (yolov8n-pose.pt or yolov8s-pose.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--project", default="runs/pose", help="Project name for saving runs")
    parser.add_argument("--name", default="volleyball", help="Run name")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}", file=sys.stderr)
        print("Run prepare_pose_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    # Use local backend model if available
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    local_model = project_root / "backend" / args.model
    if local_model.exists():
        model_path = str(local_model)
    else:
        model_path = args.model

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install ultralytics: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    print(f"Training on {args.data} for {args.epochs} epochs...")
    results = model.train(
        data=str(data_path.absolute()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    print(f"To use the fine-tuned model, set in .env:")
    print(f"  POSE_MODEL={best_weights.absolute()}")


if __name__ == "__main__":
    main()
