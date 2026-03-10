ServeSense MVP

Quick start (no Docker)

1. Backend

- cd backend
- python -m venv .venv
- . .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
- pip install -r requirements.txt
- uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

2. Frontend

- cd frontend
- npm install
- echo NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 > .env.local
- npm run dev

Open http://localhost:3000/upload

Docker

- docker compose up --build
- Frontend: http://localhost:3000
- Backend: http://localhost:8000

Flow

- Upload a video at /upload → file is proxied to backend /upload, saved under backend/uploads
- Results page navigates to /results/[id] → calls /api/analyze/[id] → backend runs analysis and returns annotated data
- Annotated images are served by backend under /uploads
