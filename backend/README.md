Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the server (auto-reload in dev):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints

- POST /upload: multipart/form-data with field `file` (mp4/mov/avi/mkv). Returns `video_id`.
- GET /analyze/{video_id}: runs basic analysis and returns JSON with annotations.

Notes

- Files are stored under `uploads/` with a UUID prefix.
- Max file size default: 100 MB. Adjust in `app/settings.py`.

