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

## Sign-in (email + password + emailed one-time code)

Every API call except `/api/health/` and the `/api/auth/*` endpoints needs a session. People sign in with an email and
password; a 6-digit code is then emailed from **pvsaketh1@gmail.com** (`EMAIL_HOST_USER`) to confirm it is them.
An unknown email is a sign-up: the account is created only after the code is verified.

- **JWT based.** `POST /api/auth/start/` returns a `challenge_token`: a signed JWT (HS256, 10 min, audience `janani:otp`)
  with claims `sub` (email), `jti` (challenge id), `pur` (login|signup) and `otp`, a *keyed HMAC* of the 6-digit code. It never
  contains the code, the password or its hash. `POST /api/auth/verify/ {challenge_token, code}` checks the JWT (signature,
  `exp`, `iss`, `aud`, pinned algorithm: `alg: none` and wrong keys are rejected) and the code, while a server-side row
  enforces single use and the 5-guess limit. Success issues a **session JWT** (audience `janani:session`, 30 days) in an
  `HttpOnly` + `Secure` + `SameSite=Lax` cookie (also accepted as `Authorization: Bearer <jwt>`). Sessions are revocable
  (logout / deactivation kill the token's `jti` immediately). A resend issues a new token and invalidates the old one.
- Codes: 6 digits, valid 10 minutes, single use, 5 wrong guesses lock them, resend cooldown 45 s.
- Throttling: failed passwords per email and per IP, codes per email per hour; `Origin` check on state-changing requests.
- Data is per account: chats, settings and the list of videos are private; a video index itself is shared and reused.
- `ALLOWED_EMAILS` / `ALLOWED_EMAIL_DOMAINS` make the site invite-only (recommended on a public URL, otherwise anyone can
  sign up and use your Groq/Gemini quota). `OWNER_EMAIL` (default `EMAIL_HOST_USER`) inherits pre-existing data.

**Email delivery (pick one; the first one configured is used):**
1. **Gmail API over HTTPS (recommended; free, sends from the real Gmail account, works on Render Free).** See below.
2. **Brevo HTTPS API** - free account, verify the sender address, set `BREVO_API_KEY`.
3. **Gmail SMTP** - `EMAIL_HOST_PASSWORD` = a Google *App Password*. Render's free web services block outbound SMTP
   (ports 25/465/587), so this needs a paid Render instance ([docs](https://render.com/docs/free)).
   Providers that offer SMTP on port 2525 (SMTP2GO, Mailjet) also work on Render Free: set `EMAIL_HOST`, `EMAIL_PORT=2525`,
   `EMAIL_HOST_USER`, `EMAIL_HOST_PASSWORD`.
`/api/health/` reports the provider (never secrets) so a broken setup is easy to spot.

### Gmail API setup (one time, about 10 minutes)
1. Go to <https://console.cloud.google.com/> and create a project (e.g. `janani`).
2. **APIs & Services > Library** > search **Gmail API** > **Enable**.
3. **APIs & Services > OAuth consent screen**: User type **External**, app name `Janani`, your Gmail as support/developer
   email. Under **Scopes** add `https://www.googleapis.com/auth/gmail.send`. Under **Audience** click **Publish app**
   (important: apps left in *Testing* get refresh tokens that expire after 7 days; an unverified published app is fine for
   your own account).
4. **APIs & Services > Credentials > Create credentials > OAuth client ID**, type **Desktop app**. Copy the client ID and secret.
5. Put these two lines in `janani/backend/.env`: `GMAIL_CLIENT_ID=...` and `GMAIL_CLIENT_SECRET=...`
6. Run `python3 janani/scripts/gmail_authorize.py`. Your browser opens, you approve *send email* for `pvsaketh1@gmail.com`
   (Google shows an "unverified app" warning: Advanced > Go to Janani). The script saves `GMAIL_REFRESH_TOKEN` into `.env`.
7. Add the same three variables (`GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`) to the Render service.

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
   `EMAIL_HOST_PASSWORD` (Gmail App Password, paid Render instance) or `BREVO_API_KEY` (free plan) for the sign-in emails,
   and optionally `ALLOWED_EMAILS`. `DJANGO_SECRET_KEY` and `DATABASE_URL` are generated/wired automatically.
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
