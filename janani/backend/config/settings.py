"""Django settings for Janani (RAG Studio). Everything is configurable via environment variables.

Runs in three places with the same code:
  * locally (`./run.sh`, Postgres on localhost, keys in backend/.env)
  * in CI (GitHub Actions Postgres service)
  * on Render (gunicorn web service + Render PostgreSQL via DATABASE_URL); the same service
    also serves the built React app (frontend/dist) through WhiteNoise
"""
import getpass
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv():
    """Tiny .env loader (backend/.env). Real environment variables win."""
    f = BASE_DIR / ".env"
    if not f.is_file():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.split(" #", 1)[0].strip().strip('"').strip("'").strip()   # drop inline comments / quotes
            os.environ.setdefault(k.strip(), v)


_load_dotenv()


def env(name, default=None):
    return os.environ.get(name, default)


def env_bool(name, default=False):
    return str(env(name, str(default))).lower() in {"1", "true", "yes", "on"}


def env_list(name, default=""):
    return [x.strip() for x in env(name, default).split(",") if x.strip()]


ON_RENDER = bool(env("RENDER"))
ON_VERCEL = bool(env("VERCEL"))
TESTING = env_bool("DJANGO_TESTING", False)

DEBUG = env_bool("DJANGO_DEBUG", not (ON_RENDER or ON_VERCEL))
SECRET_KEY = env("DJANGO_SECRET_KEY") or ("dev-only-insecure-key" if DEBUG or TESTING else None)
if not SECRET_KEY:
    raise RuntimeError("DJANGO_SECRET_KEY must be set in production (Render → service → Environment).")

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS",
                         "localhost,127.0.0.1,[::1],testserver,janani.pvsairamsaketh.in,.onrender.com")
if env("RENDER_EXTERNAL_HOSTNAME"):
    ALLOWED_HOSTS.append(env("RENDER_EXTERNAL_HOSTNAME"))

# Deployed commit (Render sets RENDER_GIT_COMMIT); CI waits until /api/health/ reports the tested commit.
APP_VERSION = (env("RENDER_GIT_COMMIT") or env("APP_VERSION") or "dev")[:7]

# Built React app served by the same service (npm run build → frontend/dist).
FRONTEND_DIST = Path(env("FRONTEND_DIST", str(BASE_DIR.parent / "frontend" / "dist")))

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "corsheaders",
    "rest_framework",
    "studio",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.middleware.common.CommonMiddleware",
    "studio.auth.AuthMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
TEMPLATES = []


# ---- PostgreSQL: DATABASE_URL (Render Postgres / any hosted Postgres) or discrete POSTGRES_* vars (local) ----
def _database():
    url = env("DATABASE_URL") or env("POSTGRES_URL")
    if url:
        u = urlparse(url)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        options = {}
        if "sslmode" in q:
            options["sslmode"] = q["sslmode"]
        elif u.hostname not in ("localhost", "127.0.0.1"):
            options["sslmode"] = "require"
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": unquote(u.path.lstrip("/")),
            "USER": unquote(u.username or ""),
            "PASSWORD": unquote(u.password or ""),
            "HOST": u.hostname,
            "PORT": str(u.port or 5432),
            "OPTIONS": options,
            # serverless: never hold connections between invocations (Neon's pooler does that)
            "CONN_MAX_AGE": 0 if ON_VERCEL else 60,
            "CONN_HEALTH_CHECKS": True,
        }
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("POSTGRES_DB", "rag_studio"),
        "USER": env("POSTGRES_USER", getpass.getuser()),
        "PASSWORD": env("POSTGRES_PASSWORD", ""),
        "HOST": env("POSTGRES_HOST", "localhost"),
        "PORT": env("POSTGRES_PORT", "5432"),
        "CONN_MAX_AGE": 60,
        "CONN_HEALTH_CHECKS": True,
    }


DATABASES = {"default": _database()}

AUTH_PASSWORD_VALIDATORS = []
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
CACHES = {"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}}

# ---- static: WhiteNoise serves the React build at the site root (/assets/*, /favicon.svg, …)
STATIC_URL = "static/"
WHITENOISE_ROOT = FRONTEND_DIST if FRONTEND_DIST.is_dir() else None
WHITENOISE_MAX_AGE = 3600
WHITENOISE_IMMUTABLE_FILE_TEST = lambda path, url: url.startswith("/assets/")  # noqa: E731  (hashed by Vite)

# ---- security (production) ----
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_SSL_REDIRECT = env_bool("SECURE_SSL_REDIRECT", True)
    SECURE_REDIRECT_EXEMPT = [r"^api/health/$"]       # Render's internal health probe uses plain HTTP
    SECURE_HSTS_SECONDS = int(env("SECURE_HSTS_SECONDS", "3600"))
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_REFERRER_POLICY = "same-origin"
    X_FRAME_OPTIONS = "DENY"

# ---- sign-in: email + password, then a one-time code emailed from EMAIL_HOST_USER ----
# Optional invite list. If either is set, only these emails / email domains can sign up or sign in.
ALLOWED_EMAILS = {e.lower() for e in env_list("ALLOWED_EMAILS")}
ALLOWED_EMAIL_DOMAINS = {d.lower().lstrip("@") for d in env_list("ALLOWED_EMAIL_DOMAINS")}

# SMTP (Gmail by default). EMAIL_HOST_PASSWORD must be a Google *App Password* (spaces are ignored).
# Render's FREE web services block outbound ports 25/465/587: use a paid instance, or BREVO_API_KEY (HTTPS) instead.
# Only this account inherits the chats / videos created before sign-in existed (never "whoever signs up first").
OWNER_EMAIL = env("OWNER_EMAIL", env("EMAIL_HOST_USER", "pvsaketh1@gmail.com")).strip().lower()
EMAIL_HOST = env("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(env("EMAIL_PORT", "587"))
EMAIL_HOST_USER = env("EMAIL_HOST_USER", "pvsaketh1@gmail.com")
EMAIL_HOST_PASSWORD = (env("EMAIL_HOST_PASSWORD") or env("SMTP_PASSWORD") or env("GMAIL_APP_PASSWORD") or "").replace(" ", "")
EMAIL_USE_SSL = env_bool("EMAIL_USE_SSL", EMAIL_PORT == 465)
EMAIL_USE_TLS = env_bool("EMAIL_USE_TLS", EMAIL_PORT == 587 and not EMAIL_USE_SSL)
EMAIL_TIMEOUT = int(env("EMAIL_TIMEOUT", "15"))
DEFAULT_FROM_EMAIL_ADDRESS = env("DEFAULT_FROM_EMAIL_ADDRESS", EMAIL_HOST_USER)
DEFAULT_FROM_EMAIL = env("DEFAULT_FROM_EMAIL", f"Janani <{DEFAULT_FROM_EMAIL_ADDRESS}>")
BREVO_API_KEY = env("BREVO_API_KEY", "")
# Gmail API over HTTPS (OAuth): sends from the real Gmail account and works on Render Free (port 443 is not blocked).
GMAIL_CLIENT_ID = env("GMAIL_CLIENT_ID", "")
GMAIL_CLIENT_SECRET = env("GMAIL_CLIENT_SECRET", "")
GMAIL_REFRESH_TOKEN = env("GMAIL_REFRESH_TOKEN", "")
_GMAIL_API_READY = bool(GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET and GMAIL_REFRESH_TOKEN)
EMAIL_PROVIDER = (env("EMAIL_PROVIDER") or ("gmail_api" if _GMAIL_API_READY else "brevo" if BREVO_API_KEY
                                            else "smtp" if EMAIL_HOST_PASSWORD else "console")).lower()
EMAIL_BACKEND = ("django.core.mail.backends.console.EmailBackend" if EMAIL_PROVIDER == "console"
                 else "django.core.mail.backends.smtp.EmailBackend")

# ---- API / CORS (the frontend is same-origin in production; CORS only matters for other setups) ----
CORS_ALLOWED_ORIGINS = env_list(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,https://janani.pvsairamsaketh.in")

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [],
    "UNAUTHENTICATED_USER": None,
    "DEFAULT_CONTENT_NEGOTIATION_CLASS": "studio.api_errors.AnyAcceptNegotiation",
    "EXCEPTION_HANDLER": "studio.api_errors.exception_handler",
    "DEFAULT_THROTTLE_CLASSES": [] if TESTING else ["rest_framework.throttling.AnonRateThrottle"],
    "DEFAULT_THROTTLE_RATES": {"anon": env("API_RATE_LIMIT", "240/min")},
}

# ---- RAG / models ----
RAG = {
    # Chat model: xAI Grok (XAI_API_KEY, xai-…) or Groq (GROQ_API_KEY, gsk_…), both OpenAI-compatible.
    "XAI_API_KEY": env("XAI_API_KEY", ""),
    "XAI_BASE_URL": env("XAI_BASE_URL", "https://api.x.ai/v1"),
    "GROK_MODEL": env("GROK_MODEL", "grok-4"),
    "GROQ_API_KEY": env("GROQ_API_KEY", ""),
    "GROQ_MODEL": env("GROQ_MODEL", ""),   # empty = pick the best chat model this account can use
    "GEMINI_API_KEY": env("GEMINI_API_KEY", "") or env("GOOGLE_API_KEY", ""),
    "GEMINI_EMBED_MODEL": env("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
    "EMBED_DIM": int(env("EMBED_DIM", "768")),
    # FAISS indexes live in Postgres; this directory is only a legacy local cache.
    "FAISS_DIR": Path(env("FAISS_DIR", str(BASE_DIR / "data" / "faiss"))),
    "DEFAULT_VIDEO": "Ilg3gGewQ5U",
    # When YouTube blocks caption downloads (common from cloud IPs such as Render's), route through a proxy.
    "YT_PROXY_URL": env("YT_PROXY_URL", ""),
    "YT_COOKIES_FILE": env("YT_COOKIES_FILE", ""),
    # Fallback when YouTube blocks caption downloads: Gemini transcribes the video's audio on Google's side.
    "GEMINI_VIDEO_MODEL": env("GEMINI_VIDEO_MODEL", ""),            # empty = pick the best flash model available
    "GEMINI_TRANSCRIBE_MAX_MIN": int(env("GEMINI_TRANSCRIBE_MAX_MIN", "120")),   # longest video portion to transcribe
    "GEMINI_TRANSCRIBE_CHUNK_MIN": int(env("GEMINI_TRANSCRIBE_CHUNK_MIN", "3")),
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"std": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "std"}},
    "root": {"handlers": ["console"], "level": env("LOG_LEVEL", "INFO")},
    "loggers": {"django.db.backends": {"level": "WARNING"}, "httpx": {"level": "WARNING"}},
}
