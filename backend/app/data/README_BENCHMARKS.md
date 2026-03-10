# Pro benchmarks

Pro benchmarks are loaded from two files (both same schema):

- **`pro_benchmarks.json`** – your saved professional benchmarks (from create-benchmark or frame analysis).
- **`literature_benchmarks.json`** – reference values from published studies; merged automatically so comparisons always have at least one benchmark per serve type (see `docs/DATASETS_RESEARCH.md`).

Each entry can have:

- `id`, `name`, `description`, `serve_type`, `camera_angle`, `source`, `level`
- `metrics`: object with **only these 4 keys** (same as analysis output):
  - `toss_contact_time_s` (seconds)
  - `contact_height_m` (meters, vertical distance from top of head to hand at hit)
  - `elbow_height_m` (meters, vertical distance from top of head to elbow at hit)
  - `toss_height_m` (meters)

## Creating benchmarks from screenshot frames

**Yes, it is possible** to derive benchmarks from screenshot frames (e.g. toss, contact, follow-through).

- The same pose estimation (YOLOv8) and ball detection can run on **static images**. You could:
  1. Upload 3 images (toss, contact, follow-through) instead of a video.
  2. Run pose + ball detection on each image.
  3. Use the **contact** frame for contact height and elbow height; the **toss** frame (or ball trajectory across frames) for toss height; and pose in the contact frame for stance width.
- **Toss → contact time** needs either:
  - Video (frame count / FPS between toss and contact), or
  - User-provided timestamps for each screenshot (e.g. "toss at 0.5s, contact at 1.2s").
- If you add an "analyze frames" flow (e.g. POST `/analyze-frames` accepting 3 images + optional timestamps), the pipeline would:
  - Estimate scale using **player height** (user input) so heights are in meters.
  - Output the same 5 metrics for use as a pro benchmark or comparison.

So: **screenshot-based benchmarks are feasible**; the main gap is an API and UI that accept multiple images and optional timestamps instead of (or in addition to) video.
