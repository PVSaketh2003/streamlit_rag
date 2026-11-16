# Janani — talk to any video, in any language

**Live:** https://janani.pvsairamsaketh.in

React + Django RAG app built from `../rag1.py`. Same retrieval pipeline and grounding prompt:
YouTube transcript → RecursiveCharacterTextSplitter (1000/200) → Gemini embeddings (normalized) → **FAISS** →
MMR (k=3, fetch_k=15, λ=0.5) → Groq / xAI Grok chat model, streamed.

| Layer | Tech |
|---|---|
| Frontend | React 18 + Vite, framer-motion, light theme, SSE streaming, ⌘K palette |
| Backend | Django 5 + DRF — runs as a Vercel Python function under `/api` |
| Database | PostgreSQL (Neon on Vercel; local Postgres in dev) — sources, chunks, chats, metrics, **and the FAISS indexes** |
| Vectors | `faiss-cpu` IndexFlatL2 + LangChain-identical MMR (`backend/studio/vstore.py`, verified against langchain-community in CI) |
| CI/CD | GitHub Actions: backend tests on Postgres → frontend tests + build → migrate Neon → deploy to Vercel → smoke test |

## Local development

```bash
cp backend/.env.example backend/.env   # add GROQ_API_KEY (or XAI_API_KEY) and GEMINI_API_KEY
./run.sh                               # Django :8000 + Vite :5173 (needs local PostgreSQL)
```

Tests: `cd backend && DJANGO_TESTING=1 python manage.py test studio` · `cd frontend && npm test`

## Deploy (one-time setup)

The pipeline (`.github/workflows/janani-ci-cd.yml`) deploys every push to `main` that passes all tests.

1. **Vercel project** — vercel.com → Add New → Project → create an empty project named `janani`
   (do **not** import the Git repo; GitHub Actions deploys it). Settings → General: Framework Preset *Other*,
   Root Directory **empty**.
2. **Database** — in the project: Storage → Create → **Neon** (Postgres) → connect to the project.
   This sets `DATABASE_URL` for the app automatically. Copy that connection string for step 4.
3. **Environment variables** (Project → Settings → Environment Variables, *Production*):
   `GROQ_API_KEY` (or `XAI_API_KEY`), `GEMINI_API_KEY`, `DJANGO_SECRET_KEY` (long random string),
   `APP_ACCESS_TOKEN` (the access code you will type in the app — strongly recommended for a public URL).
4. **GitHub secrets** (repo → Settings → Secrets and variables → Actions):
   - `VERCEL_TOKEN` — vercel.com/account/tokens
   - `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID` — Project → Settings → General (Project ID; Team/Org ID under team settings),
     or run `vercel link` once locally and read `.vercel/project.json`
   - `DATABASE_URL` — the Neon connection string (used by CI to run migrations)
5. **Domain** — Vercel project → Settings → Domains → add `janani.pvsairamsaketh.in`.
   Then in **GoDaddy** → My Products → pvsairamsaketh.in → DNS → Add record:
   `Type CNAME · Name janani · Value <the value Vercel shows, e.g. cname.vercel-dns.com> · TTL 1 hour`.
   HTTPS is issued automatically once DNS resolves (minutes, sometimes up to an hour).
6. Push to `main` (or re-run the workflow). The *deploy* job migrates Neon, deploys, and checks
   `https://janani.pvsairamsaketh.in/api/health/`.

### Notes
- **YouTube from the cloud:** YouTube often blocks transcript downloads from data-center IPs (Vercel's included).
  The app detects this and offers **Paste the transcript** (always works); `YT_PROXY_URL` routes downloads through a proxy.
- **Function limits:** answers and indexing run inside a 60 s Vercel function; Groq answers take a few seconds.
- Changing `GEMINI_EMBED_MODEL`/`EMBED_DIM` requires re-indexing sources.
