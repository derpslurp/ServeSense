# Deploying ServeSense as a Public Website

Yes, it can work as a public site. Here’s how to set it up and what to watch for.

---

## Architecture Overview

| Component | Tech | Public deployment |
|-----------|------|-------------------|
| **Frontend** | Next.js | Vercel, Netlify, or Cloudflare Pages |
| **Backend** | FastAPI + YOLO | VPS, Railway, Render, Fly.io, or similar |

The frontend talks to the backend via `NEXT_PUBLIC_BACKEND_URL`. CORS is already set to allow all origins.

---

## 1. Frontend (Next.js)

**Vercel (recommended):**

1. Push your repo to GitHub.
2. Import the project in [Vercel](https://vercel.com).
3. Set **Root Directory** to `frontend`.
4. Add environment variable:
   ```
   NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.com
   ```
5. Deploy.

**Netlify:** Same idea: root = `frontend`, add `NEXT_PUBLIC_BACKEND_URL`.

---

## 2. Backend (FastAPI + YOLO)

The backend is heavier: Python, PyTorch, Ultralytics, OpenCV. Video analysis can take 30s–2min per video.

### Option A: VPS (DigitalOcean, Linode, AWS EC2, etc.)

1. **Server:** At least 4GB RAM, 2 vCPUs. GPU optional but speeds up YOLO.
2. **Install:**
   ```bash
   sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg
   git clone https://github.com/your-repo/ServeSense.git
   cd ServeSense
   ```
3. **Backend setup:**
   ```bash
   cd backend
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Models:** Ensure `yolov8n-pose.pt`, `ball_best.pt` (if used) are in `backend/` or where the app expects them.
5. **Run with Gunicorn (production):**
   ```bash
   pip install gunicorn
   gunicorn app.main:app -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
   ```
6. **Reverse proxy:** Use Nginx or Caddy with HTTPS (e.g. Let’s Encrypt).
7. **Process manager:** Use systemd or supervisor so the backend restarts on failure.

### Option B: Railway / Render / Fly.io

These platforms can run the backend, but:

- **Railway:** Supports Python and Docker. Use a Dockerfile that includes the full project (backend + `ai/`).
- **Render:** Similar; use a Web Service with a Dockerfile.
- **Fly.io:** Good for long-running processes; video analysis fits well.

**Important:** The backend needs the `ai/` module and YOLO models. The current `backend/Dockerfile` only builds from `./backend`, so it won’t include `ai/`. You’ll need a Dockerfile that uses the project root as build context.

---

## 3. Docker Deployment (single server)

For a VPS, you can run everything with Docker. The existing `docker-compose.yml` needs a few changes for production:

1. **Frontend:** `NEXT_PUBLIC_BACKEND_URL` must be the **public** backend URL (e.g. `https://api.yoursite.com`), not `http://backend:8000`, because the browser calls the backend directly.
2. **Backend:** The image must include the `ai/` module and model files.

Example production `docker-compose.prod.yml`:

```yaml
version: '3.9'
services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - uploads_data:/app/uploads
    environment:
      - UPLOAD_DIR=/app/uploads

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        - NEXT_PUBLIC_BACKEND_URL=https://api.yoursite.com
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  uploads_data:
```

You’d need a `backend/Dockerfile` that:

- Uses `context: .` (project root)
- Copies both `backend/` and `ai/`
- Copies model files (`yolov8n-pose.pt`, `ball_best.pt`, etc.)

---

## 4. Environment Variables

| Variable | Where | Example |
|----------|-------|---------|
| `NEXT_PUBLIC_BACKEND_URL` | Frontend (build-time) | `https://api.servesense.com` |
| `UPLOAD_DIR` | Backend | `/app/uploads` (or persistent volume path) |
| `POSE_MODEL` | Backend | Path to pose model (optional) |
| `BALL_DETECT_MODEL` | Backend | Path to ball model (optional) |

---

## 5. Considerations for a Public Site

| Topic | Notes |
|-------|-------|
| **Compute** | Video analysis is CPU-heavy. Expect 30s–2min per video. A queue or “processing” state helps UX. |
| **Storage** | `uploads/` should be on a persistent volume. Ephemeral containers lose data on restart. |
| **Rate limiting** | Add rate limiting (e.g. FastAPI middleware) to avoid abuse. |
| **File size** | `MAX_FILE_SIZE_MB` (default 100) is already enforced. |
| **HTTPS** | Use HTTPS for both frontend and backend. |
| **Scaling** | For many concurrent users, run multiple backend instances behind a load balancer and use a shared storage (e.g. S3) for uploads. |

---

## 6. Quick Start: Vercel + Railway

1. **Backend on Railway:**
   - New project → Deploy from GitHub.
   - Root directory: project root (or adjust so `ai/` and `backend/` are included).
   - Use a Dockerfile that includes `ai/` and model files.
   - Set `UPLOAD_DIR` to a persistent path or use Railway volumes.
   - Note the public URL (e.g. `https://servesense-backend.up.railway.app`).

2. **Frontend on Vercel:**
   - Import repo, root = `frontend`.
   - Set `NEXT_PUBLIC_BACKEND_URL` to the Railway backend URL.
   - Deploy.

3. **CORS:** Backend already allows `*` origins, so this setup works.

---

## 7. Summary

- **Frontend:** Deploy to Vercel/Netlify with `NEXT_PUBLIC_BACKEND_URL` pointing to your backend.
- **Backend:** Deploy to a VPS or PaaS (Railway, Render, Fly.io) with enough CPU/RAM for YOLO.
- **Docker:** Use a build context that includes `ai/` and model files.
- **Storage:** Use persistent storage for `uploads/`.
- **Security:** Use HTTPS, consider rate limiting and abuse protection.
