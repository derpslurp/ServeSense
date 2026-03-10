# ServeSense scripts

## Dataset and ball detection

### 1. One-time setup (Roboflow)

1. Get an API key from [Roboflow](https://app.roboflow.com/settings/api) (free account).
2. Install the optional dependency (from repo root):
   ```bash
   pip install -r scripts/requirements-datasets.txt
   ```
   **If you get "Access is denied" on Windows** (cv2 locked): the project already has OpenCV. Install Roboflow without pulling a second OpenCV:
   ```bash
   pip install roboflow --no-deps
   ```
   Or close any running Python/backend/IDE and retry; or use `pip install -r scripts/requirements-datasets.txt --user`.
3. Set your key and download the volleyball dataset.

   **Windows (PowerShell):**
   ```powershell
   $env:ROBOFLOW_API_KEY = "your_key_here"
   python scripts/download_roboflow_volleyball.py
   ```

   **Windows (CMD):** `set ROBOFLOW_API_KEY=your_key_here` then run the script.

   **Linux/macOS:** `export ROBOFLOW_API_KEY=your_key_here` then run the script.

   You can also copy `.env.example` to `.env`, add your key there, and load it (e.g. with `python-dotenv`) if you prefer.

   Data is saved to `datasets/roboflow_volleyball/` (or pass `--out-dir`).

### 2. Evaluate ball detection

After the dataset is downloaded:

```bash
python scripts/evaluate_ball_model.py
```

This runs the same YOLOv8n model used in ServeSense on the validation set and prints metrics (e.g. mAP50). Optional:

- `--dataset-dir datasets/roboflow_volleyball` – dataset location
- `--model path/to/best.pt` – use a custom/fine-tuned model
- `--train-epochs 20` – train a ball model for 20 epochs then evaluate (saves to `runs/ball/train/`)

### 3. Train a volleyball ball model (recommended)

See **`docs/TRAIN_BALL_MODEL.md`** for the full guide. Short version:

1. Download the dataset (step 1 above).
2. Run: `python scripts/evaluate_ball_model.py --train-epochs 50`
3. Copy `runs/ball/train/weights/best.pt` to `backend/ball_best.pt`.
4. Set `BALL_DETECT_MODEL` to that path and start the backend; analyses will use your trained model for ball detection.

### 4. Serve reference images (pro comparison)

To use real volleyball images instead of placeholders in the side-by-side comparison:

```powershell
$env:ROBOFLOW_API_KEY = "your_key_here"
python scripts/download_serve_references.py
```

This downloads the Volleyball Actions dataset (CC BY 4.0), extracts toss/contact frames using pose+ball detection, saves them to `backend/uploads/benchmarks/serve_references/`, and updates `literature_benchmarks.json` and `pro_benchmarks.json`. Restart the backend to load the new images.

### 5. Literature benchmarks

No script needed. The app loads `backend/app/data/literature_benchmarks.json` automatically with pro benchmarks. See `docs/DATASETS_RESEARCH.md` for more datasets.

### 6. Fine-tune pose model on volleyball serves

Improves joint detection on blurry hit frames and volleyball-specific poses (arm extension, jump).

**Step 1: Prepare dataset** (pseudo-labeling from your serve videos or images)

```bash
# From serve videos (extracts frames, runs pose, saves labels):
python scripts/prepare_pose_dataset.py --videos serves/ --out datasets/volleyball_pose

# Or from a folder of images (e.g. after downloading Roboflow dataset):
python scripts/prepare_pose_dataset.py --images datasets/roboflow_volleyball/train/images --out datasets/volleyball_pose
```

**Step 2: Train**

```bash
python scripts/train_pose_model.py --data datasets/volleyball_pose/dataset.yaml --epochs 50
```

**Step 3: Use the fine-tuned model**

Add to `.env`:
```
POSE_MODEL=runs/pose/volleyball/weights/best.pt
```

Or copy `runs/pose/volleyball/weights/best.pt` to `backend/` and set `POSE_MODEL=backend/best.pt`.
