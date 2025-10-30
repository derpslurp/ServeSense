import os
import math
import uuid
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError(
        "mediapipe is required for analyze_video. Install with: pip install mediapipe"
    ) from exc


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class FramePose:
    frame_index: int
    visibility: float
    landmarks: List[Tuple[float, float, float]]  # x, y, z in normalized image coords


def _angle_3pts(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    denom = (np.linalg.norm(ab) * np.linalg.norm(cb))
    if denom == 0:
        return 0.0
    cosang = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _extract_pose_series(video_path: str, sample_stride: int = 2, min_visibility: float = 0.5) -> Tuple[List[FramePose], Dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    poses: List[FramePose] = []

    with mp_pose.Pose(model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_stride != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks is None:
                frame_idx += 1
                continue

            lms = result.pose_landmarks.landmark
            vis = np.mean([lm.visibility for lm in lms]) if lms else 0.0
            if vis < min_visibility:
                frame_idx += 1
                continue

            lm_list: List[Tuple[float, float, float]] = [(lm.x, lm.y, lm.z) for lm in lms]
            poses.append(FramePose(frame_index=frame_idx, visibility=float(vis), landmarks=lm_list))

            frame_idx += 1

    cap.release()

    meta = {
        "fps": float(fps),
        "total_frames": int(total_frames),
        "width": int(width),
        "height": int(height),
        "sample_stride": int(sample_stride),
    }
    return poses, meta


def _landmark_xy(landmarks: List[Tuple[float, float, float]], idx: int, w: int, h: int) -> np.ndarray:
    x, y, _ = landmarks[idx]
    return np.array([x * w, y * h], dtype=np.float32)


def _detect_key_moments(poses: List[FramePose], w: int, h: int, fps: float) -> Dict[str, int]:
    # Heuristics using right wrist and shoulder/elbow dynamics (assumes right-handed serve)
    # MediaPipe indices
    RIGHT_WRIST = 16
    RIGHT_ELBOW = 14
    RIGHT_SHOULDER = 12
    RIGHT_HIP = 24
    RIGHT_ANKLE = 28

    if not poses:
        return {}

    ys = []
    xs = []
    elbow_angles = []
    shoulder_angles = []
    ankle_y = []

    for p in poses:
        wrist = _landmark_xy(p.landmarks, RIGHT_WRIST, w, h)
        elbow = _landmark_xy(p.landmarks, RIGHT_ELBOW, w, h)
        shoulder = _landmark_xy(p.landmarks, RIGHT_SHOULDER, w, h)
        hip = _landmark_xy(p.landmarks, RIGHT_HIP, w, h)
        ankle = _landmark_xy(p.landmarks, RIGHT_ANKLE, w, h)

        ys.append(wrist[1])
        xs.append(wrist[0])
        ankle_y.append(ankle[1])
        elbow_angles.append(_angle_3pts(shoulder, elbow, wrist))
        shoulder_angles.append(_angle_3pts(elbow, shoulder, hip))

    ys = np.array(ys)
    xs = np.array(xs)
    ankle_y = np.array(ankle_y)
    elbow_angles = np.array(elbow_angles)
    shoulder_angles = np.array(shoulder_angles)

    # Smooth signals
    def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
        if len(x) < k:
            return x
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode="same")

    ys_s = smooth(ys)
    xs_s = smooth(xs)
    elbow_s = smooth(elbow_angles)
    shoulder_s = smooth(shoulder_angles)
    ankle_s = smooth(ankle_y)

    # Velocities
    vy = np.gradient(ys_s)
    vx = np.gradient(xs_s)

    # Key moment candidates
    # toss: wrist vertical velocity sign change from up to down near maximum height
    toss_idx = int(np.argmax(ys_s * -1))  # screen y increases downwards; apex is min y
    # jump: ankle y decreases (foot leaves ground) -> local minimum
    jump_idx = int(np.argmin(ankle_s)) if len(ankle_s) else toss_idx
    # hit: elbow extension near maximum + high |vx| (forward swing)
    hit_score = (180.0 - elbow_s) + (np.abs(vx) / (np.max(np.abs(vx)) + 1e-6)) * 50.0
    hit_idx = int(np.argmax(hit_score))
    # follow-through: after hit where shoulder angle stabilizes
    post = slice(min(hit_idx + 2, len(shoulder_s) - 1), len(shoulder_s))
    if post.start < len(shoulder_s):
        ft_idx_rel = int(np.argmin(np.abs(np.gradient(shoulder_s[post])))) if (len(shoulder_s[post]) > 0) else 0
        follow_idx = int(min(hit_idx + 2 + ft_idx_rel, len(shoulder_s) - 1))
    else:
        follow_idx = hit_idx

    # Map back to original frame indices
    def pose_to_frame(pose_idx: int) -> int:
        return poses[pose_idx].frame_index

    return {
        "toss": pose_to_frame(toss_idx),
        "jump": pose_to_frame(jump_idx),
        "hit": pose_to_frame(hit_idx),
        "follow_through": pose_to_frame(follow_idx),
    }


def _annotate_and_export_keyframes(video_path: str, key_frames: List[int], out_dir: str, poses_by_frame: Dict[int, FramePose]) -> List[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video for annotation")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    exported: List[Dict[str, Any]] = []
    requested = sorted(set(key_frames))
    current = 0

    for frame_idx in requested:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        pose = poses_by_frame.get(frame_idx)
        if pose:
            # Convert normalized landmarks to pixel points and draw
            landmarks_px = [(int(x * w), int(y * h)) for (x, y, z) in pose.landmarks]
            # Draw connections similar to MediaPipe default
            # Use drawing utils if available directly from results normally, but we have only coords
            # Minimal skeleton: draw points
            for (px, py) in landmarks_px:
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        name = f"kf_{frame_idx}.jpg"
        out_path = os.path.join(out_dir, name)
        cv2.imwrite(out_path, frame)
        exported.append({"frame_index": frame_idx, "image_path": out_path})

    cap.release()
    return exported


def _tips_from_moments(poses: List[FramePose], key_moments: Dict[str, int], w: int, h: int) -> List[str]:
    tips: List[str] = []
    if not poses or not key_moments:
        tips.append("Could not detect enough motion; ensure full body is visible.")
        return tips

    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    RIGHT_SHOULDER = 12
    RIGHT_HIP = 24

    def lm_xy(p: FramePose, idx: int) -> np.ndarray:
        return _landmark_xy(p.landmarks, idx, w, h)

    # Evaluate elbow extension at hit
    hit_frame = key_moments.get("hit")
    if hit_frame is not None:
        near = min(poses, key=lambda p: abs(p.frame_index - hit_frame))
        elbow = lm_xy(near, RIGHT_ELBOW)
        wrist = lm_xy(near, RIGHT_WRIST)
        shoulder = lm_xy(near, RIGHT_SHOULDER)
        angle = _angle_3pts(shoulder, elbow, wrist)
        if angle < 150:
            tips.append("Extend your hitting arm more fully at contact for a higher contact point.")

    # Torso lean at hit
    if hit_frame is not None:
        near = min(poses, key=lambda p: abs(p.frame_index - hit_frame))
        shoulder = lm_xy(near, RIGHT_SHOULDER)
        hip = lm_xy(near, RIGHT_HIP)
        torso_vec = shoulder - hip
        if torso_vec[1] > 0:  # leaning back (positive y is down)
            tips.append("Lean slightly forward at contact to drive through the ball.")

    # Toss apex timing relative to hit
    toss_frame = key_moments.get("toss")
    if toss_frame is not None and hit_frame is not None:
        dt = (hit_frame - toss_frame)
        if dt < 3:
            tips.append("Allow more time between toss apex and contact; reach at full extension.")

    if not tips:
        tips.append("Good sequence: toss, jump, hit, and follow-through detected.")
    return tips


def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Analyze a volleyball serve video.

    Returns a dictionary with keys:
    - duration_sec, fps, total_frames
    - key_moments: {toss, jump, hit, follow_through} frame indices
    - key_frames: list of {frame_index, image_path}
    - feedback: {tips: [...]} and per-frame pose snippet (subset)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    # Extract pose time series
    poses, meta = _extract_pose_series(video_path)

    # Duration estimation from frames if needed
    fps = meta.get("fps", 30.0)
    total_frames = meta.get("total_frames", 0)
    duration_sec = float(total_frames) / float(fps) if total_frames and fps else None

    # Detect key moments
    w, h = meta["width"], meta["height"]
    key_moments = _detect_key_moments(poses, w=w, h=h, fps=fps)

    # Map poses by actual frame index for quick lookup
    poses_by_frame: Dict[int, FramePose] = {p.frame_index: p for p in poses}

    # Export annotated keyframes
    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.dirname(video_path), f"{basename}_outputs_{uuid.uuid4().hex[:6]}")
    key_frames_order = [v for k, v in key_moments.items()] if key_moments else []
    exported = _annotate_and_export_keyframes(video_path, key_frames_order, out_dir, poses_by_frame)

    # Simple feedback tips
    tips = _tips_from_moments(poses, key_moments, w=w, h=h)

    # Compact pose snippet for the returned JSON (only a few joints to keep size small)
    IMPORTANT_IDX = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # shoulders/elbows/wrists/hips/knees/ankles
    pose_snippet = [
        {
            "frame_index": p.frame_index,
            "visibility": p.visibility,
            "landmarks": [p.landmarks[i] for i in IMPORTANT_IDX],
        }
        for p in poses[:: max(1, len(poses) // 50) or 1]
    ]

    return {
        "duration_sec": duration_sec,
        "fps": fps,
        "total_frames": total_frames,
        "key_moments": key_moments,
        "key_frames": exported,
        "feedback": {
            "tips": tips,
            "pose_sample": pose_snippet,
            "output_dir": out_dir,
        },
    }


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    args = parser.parse_args()
    result = analyze_video(args.video)
    print(json.dumps(result, indent=2))


