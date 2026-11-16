"""Django settings for Janani (RAG Studio). Everything is configurable via environment variables.

Runs in three places with the same code:
  * locally (`./run.sh`, Postgres on localhost, keys in backend/.env)
  * in CI (GitHub Actions Postgres service)
  * on Vercel (serverless Python function, Neon Postgres via DATABASE_URL)
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


ON_VERCEL = bool(env("VERCEL"))
TESTING = env_bool("DJANGO_TESTING", False)

DEBUG = env_bool("DJANGO_DEBUG", not ON_VERCEL)
SECRET_KEY = env("DJANGO_SECRET_KEY") or ("dev-only-insecure-key" if DEBUG or TESTING else None)
if not SECRET_KEY:
    raise RuntimeError("DJANGO_SECRET_KEY must be set in production (Vercel → Project → Settings → Environment Variables).")

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS",
                         "localhost,127.0.0.1,[::1],testserver,janani.pvsairamsaketh.in,.vercel.app")

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
    "django.middleware.common.CommonMiddleware",
    "studio.access.AccessTokenMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
TEMPLATES = []


# ---- PostgreSQL: DATABASE_URL (Neon / Vercel Postgres) or discrete POSTGRES_* vars (local) ----
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

# ---- security (production) ----
if not DEBUG:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SECURE_REFERRER_POLICY = "same-origin"
    X_FRAME_OPTIONS = "DENY"

# Optional shared access code. When set, every /api/ call (except /api/health/) must send
# header `X-Access-Token: <code>`. Strongly recommended for a public deployment: it keeps
# strangers from spending your Groq/Gemini quota or deleting your data.
APP_ACCESS_TOKEN = env("APP_ACCESS_TOKEN", "")

# ---- API / CORS (the frontend is same-origin on Vercel; CORS only matters for other setups) ----
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
    # FAISS indexes live in Postgres; this directory is only a local cache (Vercel: /tmp is the only writable dir).
    "FAISS_DIR": Path(env("FAISS_DIR", "/tmp/janani-faiss" if ON_VERCEL else str(BASE_DIR / "data" / "faiss"))),
    "DEFAULT_VIDEO": "Ilg3gGewQ5U",
    # When YouTube blocks caption downloads (common from cloud IPs such as Vercel's), route through a proxy.
    "YT_PROXY_URL": env("YT_PROXY_URL", ""),
    "YT_COOKIES_FILE": env("YT_COOKIES_FILE", ""),
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"std": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "std"}},
    "root": {"handlers": ["console"], "level": env("LOG_LEVEL", "INFO")},
    "loggers": {"django.db.backends": {"level": "WARNING"}, "httpx": {"level": "WARNING"}},
}
