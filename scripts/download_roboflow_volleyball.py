#!/usr/bin/env python3
"""
Download a Roboflow volleyball dataset in YOLO format for ball detection training/eval.

Requires: pip install roboflow
Set ROBOFLOW_API_KEY in the environment (get it from Roboflow dashboard).

Example datasets on Roboflow Universe:
  - volleyballnft/volleyball-tracker-vvvdx  (838 images, ball detection)
  - activity-graz-uni/volleyball-activity-dataset  (larger, multi-class)
  - volleyball-ai-detection/ai-volley-detection   (782 images)

Usage:
  export ROBOFLOW_API_KEY=your_key
  python scripts/download_roboflow_volleyball.py
  # Or with workspace/project/version (see Roboflow project URL):
  python scripts/download_roboflow_volleyball.py --workspace volleyballnft --project volleyball-tracker --version 1
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Roboflow volleyball dataset in YOLO format")
    parser.add_argument("--workspace", default="volleyballnft", help="Roboflow workspace name")
    parser.add_argument("--project", default="volleyball-tracker-vvvdx", help="Project name")
    parser.add_argument("--version", type=int, default=1, help="Dataset version number")
    parser.add_argument("--out-dir", default="datasets/roboflow_volleyball", help="Output directory")
    parser.add_argument("--format", default="yolov8", choices=["yolov8", "yolov5", "coco"], help="Export format")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Set ROBOFLOW_API_KEY in the environment. Get it from https://app.roboflow.com/settings/api", file=sys.stderr)
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install the Roboflow SDK: pip install roboflow", file=sys.stderr)
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    print(f"Downloading {args.project} v{args.version} as {args.format}...")
    version.download(args.format, location=args.out_dir)
    print(f"Done. Data saved to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
