Setup

1. Install dependencies:

```bash
npm install
# or
pnpm install
```

2. Configure backend URL:

Create `.env.local` with:

```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

3. Run dev server:

```bash
npm run dev
```

Routes

- /upload: upload a video; proxies to backend via /api/upload
- /results/[id]: shows annotated frames and feedback by calling backend /analyze/{id}
