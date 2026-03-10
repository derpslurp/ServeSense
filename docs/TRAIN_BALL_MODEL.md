# Best way to train the ball detector

Follow these steps to train a volleyball-specific ball model and use it in ServeSense.

---

## 1. One-time: get the dataset

1. **Roboflow API key** (free): [Roboflow → Settings → API key](https://app.roboflow.com/settings/api).

2. **Install Roboflow** (if needed):
   ```powershell
   pip install roboflow --no-deps
   ```
   (Use `--no-deps` to avoid OpenCV conflicts; the project already has OpenCV.)

3. **Download the volleyball dataset** (from repo root):
   ```powershell
   $env:ROBOFLOW_API_KEY = "your_key_here"
   python scripts/download_roboflow_volleyball.py
   ```
   Data is saved under `datasets/roboflow_volleyball/`.

---

## 2. Train the model

From the repo root:

```powershell
python scripts/evaluate_ball_model.py --train-epochs 50
```

- **50 epochs** is a good default: enough to learn volleyballs without overfitting on a small dataset. You can try 30 for a quicker run or 80 if you have a large dataset.
- Training writes checkpoints to **`runs/ball/train/`**. The best model is **`runs/ball/train/weights/best.pt`**.
- When training finishes, the script runs validation and prints mAP50 (higher = better ball detection).

---

## 3. Use the trained model in ServeSense

1. **Copy the best weights** so the backend can load them:
   ```powershell
   copy runs\ball\train\weights\best.pt backend\ball_best.pt
   ```

2. **Set the environment variable** so the analyzer uses your model instead of the default COCO “sports ball”:
   - **PowerShell (current session):**
     ```powershell
     $env:BALL_DETECT_MODEL = "C:\Users\Warren\ServeSense\backend\ball_best.pt"
     ```
     Use the full path to `ball_best.pt` (or, if you start the backend from the repo root, `backend\ball_best.pt`).
   - **Or** add to your backend `.env` (if you load env in the backend) or set it in your run script/IDE.

3. **Start the backend** (with the env var set) and run an analysis. Ball tracking will use **`ball_best.pt`** and should detect the volleyball more often, so you get more ball-based toss height, contact height, and toss→contact time.

---

## 4. Optional: re-run evaluation only

If you already trained and only want to see validation metrics:

```powershell
python scripts/evaluate_ball_model.py --model runs/ball/train/weights/best.pt
```

---

## Summary

| Step | Command / action |
|------|-------------------|
| 1. Download data | `$env:ROBOFLOW_API_KEY="..."; python scripts/download_roboflow_volleyball.py` |
| 2. Train | `python scripts/evaluate_ball_model.py --train-epochs 50` |
| 3. Copy weights | `copy runs\ball\train\weights\best.pt backend\ball_best.pt` |
| 4. Use in app | Set `BALL_DETECT_MODEL` to `backend\ball_best.pt`, then run backend and analyze a video |

Training can take from a few minutes to ~30+ minutes depending on dataset size and hardware. The first time you run the download script, it may take a couple of minutes to fetch the dataset.
