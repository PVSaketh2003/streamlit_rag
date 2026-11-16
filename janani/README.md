# Janani — talk to any video, in any language

**Live:** https://janani.pvsairamsaketh.in

React + Django RAG app built from `../rag1.py`. Same retrieval pipeline and grounding prompt:
YouTube transcript → RecursiveCharacterTextSplitter (1000/200) → Gemini embeddings (normalized) → **FAISS** →
MMR (k=3, fetch_k=15, λ=0.5) → Groq / xAI Grok chat model, streamed.

| Layer | Tech |
|---|---|
| Frontend | React 18 + Vite, framer-motion, light theme, SSE streaming, ⌘K palette |
| Backend | Django 5 + DRF on gunicorn (gthread) — one Render web service serves the API (`/api`) **and** the React build |
| Database | PostgreSQL (Render Postgres in production; local Postgres in dev) — sources, chunks, chats, metrics, **and the FAISS indexes** |
| Vectors | `faiss-cpu` IndexFlatL2 + LangChain-identical MMR (`backend/studio/vstore.py`, verified against langchain-community in CI) |
| CI/CD | GitHub Actions: backend tests on Postgres → frontend tests + build → Render deploy hook → wait until the tested commit is live |

## Local development

```bash
cp backend/.env.example backend/.env   # add GROQ_API_KEY (or XAI_API_KEY) and GEMINI_API_KEY
./run.sh                               # Django :8000 + Vite :5173 (needs local PostgreSQL)
```

Tests: `cd backend && DJANGO_TESTING=1 python manage.py test studio` · `cd frontend && npm test`

## Deploy to Render (one-time setup)

Everything is described in [`render.yaml`](../render.yaml) (a Render Blueprint): one web service `janani`
(Python 3.12, `build.sh` → `start.sh`, health check `/api/health/`) and one PostgreSQL database `janani-db`,
region Singapore. Auto-deploy is **off**: GitHub Actions deploys only commits that passed every test.

1. **Create the Blueprint** — render.com → New → **Blueprint** → connect GitHub → pick `PVSaketh2003/streamlit_rag`
   (branch `main`). Render reads `render.yaml` and creates the service + database.
2. **Fill in the secret env vars** Render asks for: `GROQ_API_KEY` (or `XAI_API_KEY`), `GEMINI_API_KEY`,
   `APP_ACCESS_TOKEN` (the access code you will type to open the app). `DJANGO_SECRET_KEY` and `DATABASE_URL`
   are generated/wired automatically.
3. **Deploy hook → GitHub** — Render → `janani` service → Settings → **Deploy Hook** → copy the URL →
   GitHub repo → Settings → Secrets and variables → Actions → New secret `RENDER_DEPLOY_HOOK_URL`.
   If Render gave the service a different address than `https://janani.onrender.com`, also add a repository
   **variable** `RENDER_URL` with that address.
4. **Domain** — `render.yaml` already attaches `janani.pvsairamsaketh.in`. In **GoDaddy** → My Products →
   pvsairamsaketh.in → DNS → Add record: `Type CNAME · Name janani · Value <your-service>.onrender.com · TTL 1 hour`.
   Render then verifies it and issues HTTPS automatically (minutes, sometimes up to an hour).
5. **Ship** — merge to `main` (or run the workflow manually). The *deploy* job triggers Render and waits until
   `/api/health/` reports the exact commit that was tested, with the database connected.

Free-plan caveats: the free web instance sleeps after ~15 min idle (first request then takes ~30–60 s), and Render's
free PostgreSQL **expires after 30 days** — switch `plan:` in `render.yaml` (e.g. `starter` / `basic-256mb`) to keep data.

### Notes
- **YouTube from the cloud:** YouTube often blocks caption downloads from data-center IPs (Render's included).
  Janani tries, in order: cached transcript (Postgres) → `youtube-transcript-api` → `yt-dlp` (several player clients) →
  **Gemini transcribes the video's audio on Google's servers** (the request never touches YouTube from our IP; public videos only;
  10-minute windows, up to `GEMINI_TRANSCRIBE_MAX_MIN`=120 minutes per video, saved so it happens once) → **Paste the transcript**.
  Optional: `YT_PROXY_URL` routes downloads through a proxy; `GEMINI_VIDEO_MODEL` pins the transcription model
  (default: newest flash model your key can use, with automatic fallback when a model is overloaded or retired).
- Answers stream live over Server-Sent Events (gunicorn gthread workers, 180 s request timeout).
- Changing `GEMINI_EMBED_MODEL`/`EMBED_DIM` requires re-indexing sources.
