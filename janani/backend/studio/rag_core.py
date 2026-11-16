"""RAG pipeline ported from rag1.py (Streamlit removed, retrieval logic unchanged).

  YouTubeTranscriptApi -> RecursiveCharacterTextSplitter(1000/200)
  -> Gemini embeddings (normalized) -> FAISS
  -> MMR retrieval (k=3, fetch_k=15, lambda_mult=0.5)
  -> Grok (xAI, OpenAI-compatible API, streaming)
"""
import gc
import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from django.conf import settings

log = logging.getLogger("studio.rag")

CFG = settings.RAG
def _resolve_llm():
    """Pick the OpenAI-compatible chat provider from the configured key.
    xAI keys start with 'xai-'; Groq keys start with 'gsk_' (an easy mix-up: Grok vs Groq)."""
    xai, groq = CFG["XAI_API_KEY"].strip(), CFG.get("GROQ_API_KEY", "").strip()
    if groq or xai.startswith("gsk_"):
        return {"provider": "Groq", "short": "Groq", "key": groq or xai, "env": "GROQ_API_KEY" if groq else "XAI_API_KEY",
                "base_url": "https://api.groq.com/openai/v1", "model": CFG.get("GROQ_MODEL") or "", "auto": not CFG.get("GROQ_MODEL")}
    return {"provider": "xAI Grok", "short": "Grok", "key": xai, "env": "XAI_API_KEY",
            "base_url": CFG["XAI_BASE_URL"], "model": CFG["GROK_MODEL"]}


LLM = _resolve_llm()
XAI_API_KEY = LLM["key"]
XAI_BASE_URL = LLM["base_url"]
GROK_MODEL = LLM["model"]
GEMINI_API_KEY = CFG["GEMINI_API_KEY"]
GEMINI_EMBED_MODEL = CFG["GEMINI_EMBED_MODEL"]
EMBED_DIM = CFG["EMBED_DIM"]
EMBED_TAG = f"{GEMINI_EMBED_MODEL}@{EMBED_DIM}"
FAISS_DIR: Path = CFG["FAISS_DIR"]
DEFAULT_VIDEO = CFG["DEFAULT_VIDEO"]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 3
FETCH_K = 15
LAMBDA_MULT = 0.5
TEMPERATURE = 0.5

PROMPT_TEMPLATE = """You are a helpful AI assistant.
Answer the question ONLY using the provided transcript context below.
Do not assume or extrapolate any facts not directly mentioned in the text.
If the context is insufficient to answer the question, just say "I don't know."
{language_rules}
Transcript Context:
---
{context}
---

Question: {question}
Answer:"""


def format_prompt(context, question, language_rules):
    return PROMPT_TEMPLATE.format(context=context, question=question, language_rules=language_rules)


# ---------------------------------------------------------------- video ids

VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
            "youtube-nocookie.com", "www.youtube-nocookie.com"}
YT_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}
YT_PATH_PREFIXES = {"shorts", "embed", "live", "v"}


def extract_youtube_video_id(value):
    text = (value or "").strip()
    if not text:
        raise ValueError("Paste a YouTube video link to get started.")
    if VIDEO_ID_RE.fullmatch(text):
        return text
    candidate = text if "://" in text else "https://" + text
    try:
        parsed = urlparse(candidate)
        host = (parsed.hostname or "").lower()
    except ValueError:
        raise ValueError("That link is not a valid URL. Paste the full YouTube video link.") from None
    if not host or "." not in host:
        raise ValueError("That doesn't look like a YouTube link or an 11-character video ID.")
    parts = [seg for seg in parsed.path.split("/") if seg]
    if host in YT_SHORT_HOSTS:
        video_id = parts[0] if parts else ""
    elif host in YT_HOSTS:
        if parts and parts[0] == "watch":
            video_id = (parse_qs(parsed.query).get("v") or [""])[0].strip()
        elif len(parts) >= 2 and parts[0] in YT_PATH_PREFIXES:
            video_id = parts[1]
        elif parts and parts[0] == "playlist":
            raise ValueError("That is a playlist link. Open one video and paste its own link "
                             "(it contains ?v=…). Playlists are not indexed.")
        else:
            raise ValueError("That YouTube link doesn't point to a single video. "
                             "Open the video and paste its link (it contains ?v=…).")
    else:
        raise ValueError(f"Unsupported website ({host}). Only YouTube video links are supported.")
    if not video_id:
        raise ValueError("No video ID found in that link. Open the video and copy its full link.")
    if not VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError(f"'{video_id[:20]}' is not a valid YouTube video ID (it must be 11 characters: "
                         "letters, digits, - and _).")
    return video_id


# ---------------------------------------------------------------- languages

LANGS = [
    ("en", "English", "English", "Latin"),
    ("te", "Telugu", "తెలుగు", "Telugu"),
    ("hi", "Hindi", "हिन्दी", "Devanagari"),
    ("ta", "Tamil", "தமிழ்", "Tamil"),
    ("kn", "Kannada", "ಕನ್ನಡ", "Kannada"),
    ("ml", "Malayalam", "മലയാളം", "Malayalam"),
    ("mr", "Marathi", "मराठी", "Devanagari"),
    ("bn", "Bengali", "বাংলা", "Bengali"),
    ("gu", "Gujarati", "ગુજરાતી", "Gujarati"),
    ("pa", "Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi"),
    ("ur", "Urdu", "اردو", "Arabic"),
    ("es", "Spanish", "Español", "Latin"),
    ("fr", "French", "Français", "Latin"),
    ("de", "German", "Deutsch", "Latin"),
    ("ja", "Japanese", "日本語", "Japanese"),
    ("ko", "Korean", "한국어", "Korean"),
    ("zh", "Chinese", "中文", "Chinese"),
    ("ar", "Arabic", "العربية", "Arabic"),
    ("pt", "Portuguese", "Português", "Latin"),
    ("ru", "Russian", "Русский", "Cyrillic"),
]
LANG_BY_CODE = {c: {"code": c, "name": n, "native": nat, "script": sc} for c, n, nat, sc in LANGS}
EXTRA_SCRIPTS = {"fa": "Arabic", "ps": "Arabic", "uk": "Cyrillic", "bg": "Cyrillic", "sr": "Cyrillic"}
_SCRIPT_RANGES = [
    ("Telugu", 0x0C00, 0x0C7F), ("Kannada", 0x0C80, 0x0CFF), ("Tamil", 0x0B80, 0x0BFF),
    ("Malayalam", 0x0D00, 0x0D7F), ("Bengali", 0x0980, 0x09FF), ("Gujarati", 0x0A80, 0x0AFF),
    ("Gurmukhi", 0x0A00, 0x0A7F), ("Devanagari", 0x0900, 0x097F), ("Arabic", 0x0600, 0x06FF),
    ("Arabic", 0x0750, 0x077F), ("Kana", 0x3040, 0x30FF), ("Korean", 0xAC00, 0xD7AF),
    ("Han", 0x4E00, 0x9FFF), ("Cyrillic", 0x0400, 0x04FF),
]
LOW_RESOURCE_SCRIPTS = {"Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Gujarati", "Gurmukhi", "Arabic"}


def lang_info(code, fallback_name=None):
    base = (code or "").split("-")[0].lower()
    if base in LANG_BY_CODE:
        return LANG_BY_CODE[base]
    name = fallback_name or base or "Unknown"
    return {"code": base, "name": name, "native": name, "script": EXTRA_SCRIPTS.get(base, "Latin")}


def detect_script(text):
    counts = {}
    for ch in text:
        if not ch.isalpha():
            continue
        o = ord(ch)
        label = None
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= o <= hi:
                label = name
                break
        if label is None:
            label = "Latin" if o < 0x250 else "Other"
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "Latin"
    if "Kana" in counts:
        return "Japanese"
    if "Han" in counts and counts["Han"] >= max(counts.values()):
        return "Chinese"
    top = max(counts, key=counts.get)
    return "Japanese" if top == "Kana" else top


class LanguageUnavailable(Exception):
    def __init__(self, requested, tracks):
        self.requested = requested
        self.tracks = tracks
        name = lang_info(requested)["name"]
        avail = ", ".join(sorted({t["name"] for t in tracks}))
        super().__init__(f"This video has no {name} captions. Available caption languages: {avail}.")


def index_key(video_id, lang_code):
    return f"{video_id}:{lang_code}"


def pick_track(tracks, requested):
    def best(matches):
        manual = [t for t in matches if not t["generated"]]
        return (manual or matches)[0]

    if requested in (None, "auto", "other"):
        generated = [t for t in tracks if t["generated"]]
        if generated:
            code = generated[0]["code"]
            return best([t for t in tracks if t["code"] == code]), "spoken language detected from YouTube's auto-generated captions"
        for pref in ("en", "hi"):
            m = [t for t in tracks if t["code"].split("-")[0].lower() == pref]
            if m:
                return best(m), "no auto-generated track, so the preferred available captions were used"
        return tracks[0], "first available captions"
    matches = [t for t in tracks if t["code"] == requested] or \
              [t for t in tracks if t["code"].split("-")[0].lower() == requested]
    if matches:
        return best(matches), "selected by you"
    raise LanguageUnavailable(requested, tracks)


def language_rules(answer_code, video_lang_name):
    a = lang_info(answer_code)
    lines = [
        f"The transcript context is in {video_lang_name}. The question may be written in any language.",
        f"Write your entire answer in {a['name']} ({a['native']}), in natural, fluent {a['name']} using its standard script. "
        f"Do not answer in any other language, and do not confuse {a['name']} with related languages.",
        "Convey the meaning of the transcript faithfully. This is translation of the context, not summarization, "
        "and you must not add information that is not in the context.",
        "Keep technical terms, code, formulas and product names (for example Machine Learning, Stochastic Gradient "
        "Descent, TensorFlow, PyTorch) accurate: keep them in their original form or add them in Latin letters next to "
        "the explanation. Do not mix languages unnecessarily.",
    ]
    if answer_code == "en":
        lines.append('If the context is insufficient, just say "I don\'t know."')
    else:
        lines.append(f'If the context is insufficient, say "I don\'t know." written in {a["name"]}.')
    return "\n" + "\n".join(lines) + "\n"


def split_reasoning(raw):
    """Separate Qwen3 <think> reasoning from the final answer."""
    if "</think>" in raw:
        before, after = raw.split("</think>", 1)
        return before.replace("<think>", "").strip(), after.strip(), False
    if raw.lstrip().startswith("<think>"):
        return raw.lstrip()[len("<think>"):].strip(), "", True
    return "", raw, False


# ---------------------------------------------------------------- runtime state

class ConfigError(ValueError):
    """Missing/invalid provider configuration (shown to the user verbatim)."""


_state_lock = threading.Lock()
_state = {"embed_ok_s": None, "llm_ok_s": None}
_building = set()
_indexes = {}  # source_id -> {"vs", "emb"}
_clients = {}


def runtime_status():
    return {
        "embedding_ready": bool(GEMINI_API_KEY),
        "llm_ready": bool(XAI_API_KEY),
        "embedding_verified": _state["embed_ok_s"] is not None,
        "llm_verified": _state["llm_ok_s"] is not None,
        "llm_model": GROK_MODEL,
        "llm_provider": LLM["provider"],
        "llm_short": LLM["short"],
        "embedding_model": GEMINI_EMBED_MODEL,
        "indexes_in_memory": len(_indexes),
        "streaming": True,
    }


# Preferred chat models, best first. Qwen models are deliberately excluded.
PREFERRED_MODELS = [
    "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "meta-llama/llama-4-maverick-17b-128e-instruct",
    "moonshotai/kimi-k2-instruct", "openai/gpt-oss-20b", "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant", "grok-4", "grok-3", "grok-3-mini",
]
NON_CHAT = ("whisper", "guard", "orpheus", "tts", "embed", "safeguard")


def _ensure_model(client):
    """Make sure GROK_MODEL is something this account can actually call; otherwise pick the best one."""
    global GROK_MODEL
    if _state.get("model_checked"):
        return
    try:
        available = [m.id for m in client.models.list().data]
    except Exception as exc:
        log.warning("Could not list models (%s); using %s", type(exc).__name__, GROK_MODEL or "default")
        available = []
    if available and GROK_MODEL not in available:
        chat = [m for m in available if not any(w in m.lower() for w in NON_CHAT) and "qwen" not in m.lower()]
        pick = next((m for m in PREFERRED_MODELS if m in chat), chat[0] if chat else None)
        if pick:
            if GROK_MODEL:
                log.warning("Model %s is not available on this %s account; using %s", GROK_MODEL, LLM["provider"], pick)
            GROK_MODEL = pick
    if not GROK_MODEL:
        GROK_MODEL = "openai/gpt-oss-120b" if LLM["short"] == "Groq" else "grok-4"
    LLM["model"] = GROK_MODEL
    log.info("Chat model: %s via %s", GROK_MODEL, LLM["provider"])
    _state["model_checked"] = True


def _grok_client():
    if not XAI_API_KEY:
        raise ConfigError("No chat-model key is set. Add XAI_API_KEY (xAI Grok) or GROQ_API_KEY (Groq) "
                          "to backend/.env and restart the backend.")
    if "grok" not in _clients:
        from openai import OpenAI
        _clients["grok"] = OpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL, timeout=120, max_retries=2)
    _ensure_model(_clients["grok"])
    return _clients["grok"]


def _gemini_client():
    if not GEMINI_API_KEY:
        raise ConfigError("GEMINI_API_KEY is not set. Add it to backend/.env and restart the backend.")
    if "gemini" not in _clients:
        from google import genai
        _clients["gemini"] = genai.Client(api_key=GEMINI_API_KEY)
    return _clients["gemini"]


def _embed_batch(texts, task):
    from google.genai import types
    import numpy as np
    client = _gemini_client()
    last = None
    for attempt in range(4):
        try:
            res = client.models.embed_content(
                model=GEMINI_EMBED_MODEL, contents=texts,
                config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIM))
            vecs = np.array([e.values for e in res.embeddings], dtype="float32")
            _state["embed_ok_s"] = _state["embed_ok_s"] or 0.001
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return (vecs / np.clip(norms, 1e-12, None)).tolist()    # truncated Gemini vectors are not unit-norm
        except Exception as exc:
            last = exc
            if attempt == 3 or not any(m in str(exc) for m in ("429", "500", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED")):
                break
            time.sleep(1.5 * (2 ** attempt))
    raise last


class TimedEmbeddings:
    """Gemini embeddings (langchain Embeddings interface) that also measure time spent."""

    def __new__(cls):
        from langchain_core.embeddings import Embeddings

        class _Gemini(Embeddings):
            def __init__(self):
                self.doc_s, self.query_s = 0.0, 0.0

            def embed_documents(self, texts):
                t = time.perf_counter()
                out = []
                for i in range(0, len(texts), 50):
                    out += _embed_batch(texts[i:i + 50], "RETRIEVAL_DOCUMENT")
                self.doc_s += time.perf_counter() - t
                return out

            def embed_query(self, text):
                t = time.perf_counter()
                out = _embed_batch([text], "RETRIEVAL_QUERY")[0]
                self.query_s += time.perf_counter() - t
                return out

        return _Gemini()


def check_llm():
    """Cheap round-trip proving the Grok key/model work. Returns seconds."""
    t = time.perf_counter()
    _chat_create([{"role": "user", "content": "Reply with: ok"}], max_tokens=64)
    _state["llm_ok_s"] = time.perf_counter() - t
    return _state["llm_ok_s"]


def check_embeddings():
    t = time.perf_counter()
    _embed_batch(["ping"], "RETRIEVAL_QUERY")
    _state["embed_ok_s"] = time.perf_counter() - t
    return _state["embed_ok_s"]


# ---------------------------------------------------------------- transcripts
#
# Order of attempts (each step only runs if the previous one could not deliver):
#   1. Postgres cache (TranscriptCache / TrackCache) — anything fetched once is never fetched again
#   2. youtube-transcript-api (optionally through YT_PROXY_URL), with short retries on transient errors
#   3. yt-dlp, if installed (different YouTube endpoint; also honours YT_PROXY_URL / YT_COOKIES_FILE)
#   4. the user pastes the transcript (POST /api/sources/manual/)

YT_PROXY_URL = CFG.get("YT_PROXY_URL", "")
YT_COOKIES_FILE = CFG.get("YT_COOKIES_FILE", "")
BLOCKED_ERRORS = {"IpBlocked", "RequestBlocked", "PoTokenRequired"}
TRANSIENT_ERRORS = {"YouTubeRequestFailed", "ConnectionError", "ReadTimeout", "ConnectTimeout", "ChunkedEncodingError"}


class YouTubeBlocked(Exception):
    """YouTube refused every route we tried. The UI offers 'paste the transcript' instead."""

    def __init__(self, detail=""):
        msg = ("YouTube is blocking caption requests from this network right now, so the transcript could not be "
               "downloaded. Paste the transcript instead, wait a while and retry, or set YT_PROXY_URL in backend/.env.")
        super().__init__(msg)
        self.detail = detail


def yt_dlp_available():
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return False


def _yt_api():
    from youtube_transcript_api import YouTubeTranscriptApi
    if YT_PROXY_URL:
        from youtube_transcript_api.proxies import GenericProxyConfig
        return YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=YT_PROXY_URL, https_url=YT_PROXY_URL))
    return YouTubeTranscriptApi()


def _with_retries(fn, attempts=3):
    """Retry transient network failures quickly. IP blocks are not retried here (they don't clear in seconds)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            name = type(exc).__name__
            if name in TRANSIENT_ERRORS and i < attempts - 1:
                time.sleep(1.2 * (2 ** i))
                continue
            raise


def _ydl(extra=None):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True, "socket_timeout": 15}
    if YT_PROXY_URL:
        opts["proxy"] = YT_PROXY_URL
    if YT_COOKIES_FILE and os.path.isfile(YT_COOKIES_FILE):
        opts["cookiefile"] = YT_COOKIES_FILE
    opts.update(extra or {})
    return yt_dlp.YoutubeDL(opts)


def _ydl_info(video_id):
    with _ydl() as ydl:
        return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)


def _ydl_tracks(video_id):
    info = _ydl_info(video_id)
    tracks, seen = [], set()
    for kind, generated in (("subtitles", False), ("automatic_captions", True)):
        for code, fmts in (info.get(kind) or {}).items():
            if code == "live_chat" or (code, generated) in seen or not fmts:
                continue
            # automatic_captions also lists every machine-translated target; keep only the original track(s)
            if generated and "-orig" not in code and info.get("language") and code.split("-")[0] != info["language"].split("-")[0]:
                continue
            seen.add((code, generated))
            name = (fmts[0].get("name") or lang_info(code)["name"]).replace(" (auto-generated)", "").replace(" (Original)", "")
            tracks.append({"code": code.replace("-orig", ""), "name": name, "generated": generated, "translatable": True})
    return tracks


def _ydl_transcript(video_id, lang_code):
    """Download one caption track via yt-dlp and flatten it to plain text."""
    info = _ydl_info(video_id)
    for kind in ("subtitles", "automatic_captions"):
        pool = info.get(kind) or {}
        for code in (lang_code, f"{lang_code}-orig"):
            fmts = pool.get(code)
            if not fmts:
                continue
            fmt = next((f for f in fmts if f.get("ext") == "json3"), None) or next((f for f in fmts if f.get("ext") == "vtt"), None)
            if not fmt:
                continue
            with _ydl() as ydl:
                raw = ydl.urlopen(fmt["url"]).read().decode("utf-8", "replace")
            if fmt["ext"] == "json3":
                data = json.loads(raw)
                parts = ["".join(seg.get("utf8", "") for seg in ev.get("segs") or []) for ev in data.get("events", [])]
            else:
                parts = [ln for ln in raw.splitlines()
                         if ln.strip() and "-->" not in ln and not ln.startswith(("WEBVTT", "Kind:", "Language:", "NOTE"))
                         and not ln.strip().isdigit()]
                parts = [re.sub(r"<[^>]+>", "", p) for p in parts]
                parts = [p for i, p in enumerate(parts) if i == 0 or p != parts[i - 1]]   # vtt repeats rolling lines
            text = re.sub(r"\s+", " ", " ".join(p.replace("\n", " ") for p in parts)).strip()
            if text:
                return text
    raise ValueError(f"No {lang_info(lang_code)['name']} captions found via yt-dlp.")


def _db_transcript(video_id, lang_code):
    from .models import TranscriptCache
    row = TranscriptCache.objects.filter(video_id=video_id, lang_code=lang_code).first()
    return row.text if row else None


def save_transcript(video_id, lang_code, text, origin):
    from .models import TranscriptCache
    TranscriptCache.objects.update_or_create(video_id=video_id, lang_code=lang_code,
                                             defaults={"text": text, "origin": origin})


def load_transcript(video_id, lang_code="en"):
    """Returns (text, was_cached, origin)."""
    cached = _db_transcript(video_id, lang_code)
    if cached:
        return cached, True, "cache"
    errors = []
    try:
        fetched = _with_retries(lambda: _yt_api().fetch(video_id, languages=[lang_code]))
        text = " ".join(chunk.text for chunk in fetched)
        save_transcript(video_id, lang_code, text, "youtube")
        return text, False, "youtube"
    except Exception as exc:
        name = type(exc).__name__
        errors.append(name)
        if name not in BLOCKED_ERRORS | TRANSIENT_ERRORS:
            raise                      # real answers like "captions disabled" or "no such language"
        log.warning("youtube-transcript-api failed for %s/%s: %s", video_id, lang_code, name)
    if yt_dlp_available():
        try:
            text = _ydl_transcript(video_id, lang_code)
            save_transcript(video_id, lang_code, text, "yt-dlp")
            log.info("Transcript for %s/%s fetched via yt-dlp fallback", video_id, lang_code)
            return text, False, "yt-dlp"
        except Exception as exc:
            errors.append(f"yt-dlp: {type(exc).__name__}")
            log.warning("yt-dlp fallback failed for %s/%s: %s", video_id, lang_code, exc)
    raise YouTubeBlocked(", ".join(errors))


def list_transcript_tracks(video_id):
    from .models import TrackCache
    row = TrackCache.objects.filter(video_id=video_id).first()
    if row and row.tracks:
        return row.tracks
    try:
        tracks = []
        for t in _with_retries(lambda: _yt_api().list(video_id)):
            tracks.append({
                "code": t.language_code,
                "name": str(t.language).replace(" (auto-generated)", ""),
                "generated": bool(t.is_generated),
                "translatable": bool(t.is_translatable),
            })
    except Exception as exc:
        name = type(exc).__name__
        if name not in BLOCKED_ERRORS | TRANSIENT_ERRORS:
            raise
        log.warning("Caption listing blocked for %s (%s)", video_id, name)
        tracks = None
        if yt_dlp_available():
            try:
                tracks = _ydl_tracks(video_id)
            except Exception as exc2:
                log.warning("yt-dlp track listing failed for %s: %s", video_id, exc2)
        if not tracks:
            raise YouTubeBlocked(name) from None
    if not tracks:
        raise ValueError("No caption tracks were found for this video.")
    TrackCache.objects.update_or_create(video_id=video_id, defaults={"tracks": tracks})
    return tracks


def clear_transcript_cache():
    from .models import TrackCache, TranscriptCache
    TranscriptCache.objects.all().delete()
    TrackCache.objects.all().delete()
    fetch_video_title.cache_clear()


@lru_cache(maxsize=256)
def fetch_video_title(video_id):
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = Request(url, headers={"User-Agent": "mlx-rag-studio"})
        with urlopen(req, timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8")).get("title")
    except Exception as exc:
        log.info("Title lookup skipped for %s (%s)", video_id, type(exc).__name__)
        return None


# ---------------------------------------------------------------- indexing

def building_keys():
    return set(_building)


def build_index(video_id, track, reason, tracks, transcript_text=None):
    """Yields ('stage', key, status, ms, detail) and finally ('done', data).
    `data` holds everything needed to persist a VideoSource (+ the live FAISS store)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from .vstore import VectorIndex

    code = track["code"]
    key = index_key(video_id, code)
    info = lang_info(code, track["name"])
    timings, details = {}, {}
    cur = "fetch"
    with _state_lock:
        if key in _building:
            raise ValueError("This video is already being indexed. Wait for it to finish.")
        _building.add(key)
    try:
        yield ("stage", "fetch", "running", None, f"{info['name']} captions")
        t = time.perf_counter()
        if transcript_text is not None:
            origin = "pasted by you"
        else:
            transcript_text, was_cached, origin = load_transcript(video_id, code)
            origin = {"cache": "from cache", "youtube": "downloaded", "yt-dlp": "downloaded via yt-dlp"}[origin]
        timings["fetch"] = time.perf_counter() - t
        if not transcript_text.strip():
            raise ValueError("The transcript is empty.")
        details["fetch"] = f"{len(transcript_text):,} chars · {info['name']} · {origin}"
        yield ("stage", "fetch", "done", timings["fetch"] * 1000, details["fetch"])

        cur = "split"
        yield ("stage", "split", "running", None, "")
        t = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.create_documents([transcript_text])
        timings["split"] = time.perf_counter() - t
        if not chunks:
            raise ValueError("Splitting produced no chunks.")
        details["split"] = f"{len(chunks):,} chunks · {CHUNK_SIZE}/{CHUNK_OVERLAP}"
        yield ("stage", "split", "done", timings["split"] * 1000, details["split"])

        cur = "embed"
        yield ("stage", "embed", "running", None, f"{GEMINI_EMBED_MODEL} · {len(chunks)} chunks")
        _gemini_client()   # fail fast with a clear message if the key is missing
        emb = TimedEmbeddings()
        t = time.perf_counter()
        texts = [c.page_content for c in chunks]
        vector_store = VectorIndex.build(emb.embed_documents(texts), texts)
        total_build = time.perf_counter() - t
        timings["embed"] = emb.doc_s
        timings["faiss"] = max(total_build - emb.doc_s, 0.0)
        details["embed"] = f"{len(chunks):,} vectors · {GEMINI_EMBED_MODEL}"
        details["faiss"] = f"{vector_store.ntotal:,} vectors · dim {vector_store.dim}"
        yield ("stage", "embed", "done", timings["embed"] * 1000, details["embed"])
        cur = "faiss"
        yield ("stage", "faiss", "done", timings["faiss"] * 1000, details["faiss"])

        index_file = f"{video_id}_{code}.index".replace("/", "_")
        t = time.perf_counter()
        index_blob = vector_store.to_bytes()          # stored in Postgres with the source
        timings["faiss"] += time.perf_counter() - t

        yield ("done", {
            "video_id": video_id, "key": key, "title": fetch_video_title(video_id) or "",
            "lang_code": code, "lang_name": info["name"],
            "lang_kind": "auto-generated" if track["generated"] else "manual",
            "lang_reason": reason, "tracks": tracks,
            "n_chunks": len(chunks), "n_chars": len(transcript_text), "n_words": len(transcript_text.split()),
            "timings": timings, "details": details, "total_s": sum(timings.values()),
            "index_file": index_file, "embed_model": EMBED_TAG, "index_blob": index_blob,
            "chunk_texts": texts,
            "_vs": vector_store, "_emb": emb,
        })
    except Exception as exc:
        yield ("stage", cur, "error", None, str(exc)[:160])
        raise
    finally:
        with _state_lock:
            _building.discard(key)


def register_index(source_id, vs, emb):
    _indexes[source_id] = {"vs": vs, "emb": emb}


def drop_index(source_id, index_file=None):
    _indexes.pop(source_id, None)
    if index_file:
        try:
            (FAISS_DIR / index_file).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Could not delete %s: %s", index_file, exc)
    gc.collect()


def get_index(source):
    """Return the live {'vs','emb'} for a VideoSource. After a restart (or on a fresh serverless
    instance) the FAISS index is rebuilt from the bytes stored in Postgres; older local sources
    fall back to the index file on disk."""
    entry = _indexes.get(source.id)
    if entry:
        return entry
    from .vstore import VectorIndex

    if source.embed_model != EMBED_TAG:
        raise ValueError("This index was built with a different embedding model. "
                         "Remove the source in Sources and index the video again.")
    texts = list(source.chunks.order_by("idx").values_list("text", flat=True))
    if source.index_blob:
        vs = VectorIndex.from_bytes(source.index_blob, texts)
    else:
        path = FAISS_DIR / source.index_file
        if not path.is_file():
            raise FileNotFoundError("The saved FAISS index is missing. Remove this source and index the video again.")
        import faiss
        vs = VectorIndex(faiss.read_index(str(path)), texts)
    register_index(source.id, vs, TimedEmbeddings())
    return _indexes[source.id]


# ---------------------------------------------------------------- generation

_param_variant = {"i": 0}


def _chat_create(messages, max_tokens, temperature=None, stream=False):
    """Call Grok, adapting to what the chosen model accepts. Reasoning models (e.g. grok-4) reject some
    sampling / token-limit / stream_options parameters with HTTP 400, so we retry with simpler sets
    and remember the first one that works."""
    from openai import BadRequestError, NotFoundError
    client = _grok_client()          # resolves GROK_MODEL first
    effort = {"reasoning_effort": "low"} if "gpt-oss" in GROK_MODEL else {}
    variants = [
        lambda: dict(temperature=temperature, max_tokens=max_tokens, stream_options={"include_usage": True}, **effort),
        lambda: dict(temperature=temperature, max_tokens=max_tokens, stream_options={"include_usage": True}),
        lambda: dict(temperature=temperature, max_completion_tokens=max_tokens, stream_options={"include_usage": True}),
        lambda: dict(max_completion_tokens=max_tokens, stream_options={"include_usage": True}),
        lambda: dict(max_completion_tokens=max_tokens),
        lambda: dict(),
    ]
    last = None
    for i in range(_param_variant["i"], len(variants)):
        kw = {k: v for k, v in variants[i]().items() if v is not None}
        if not stream:
            kw.pop("stream_options", None)
        try:
            res = client.chat.completions.create(model=GROK_MODEL, messages=messages, stream=stream, **kw)
            _param_variant["i"] = i
            return res
        except (BadRequestError, NotFoundError) as exc:
            last = exc
            msg = _provider_message(exc)
            if _is_fatal(msg):
                log.error("%s rejected the request: %s", LLM["provider"], msg)
                raise
            log.warning("%s rejected request variant %d (%s); trying a simpler one", LLM["provider"], i, msg)
    raise last


def _is_fatal(msg):
    m = msg.lower()
    return any(w in m for w in ("api key", "api_key", "unauthor", "permission", "credit", "model", "not found",
                                "does not exist", "quota", "billing", "blocked", "suspended"))


def _provider_message(exc):
    """Best human-readable error text from an OpenAI-SDK error. Providers differ: OpenAI sends
    {"error": {"message": ...}}, xAI sends {"code": ..., "error": "<text>"}, and the SDK may
    hand us the dict, just the inner string, or nothing — so fall back to the raw response."""
    body = getattr(exc, "body", None)
    if isinstance(body, str) and body.strip():
        return body.strip()[:400]
    if isinstance(body, dict):
        err = body.get("error", body)
        msg = err.get("message") if isinstance(err, dict) else err
        msg = msg or body.get("message")
        if msg:
            return str(msg)[:400]
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            data = resp.json()
            err = data.get("error", data)
            msg = err.get("message") if isinstance(err, dict) else err
            if msg:
                return str(msg)[:400]
        except Exception:
            text = (getattr(resp, "text", "") or "").strip()
            if text:
                return text[:400]
    return str(exc)[:400]


def bridge_query(question):
    """Translate a short question to English with Grok so retrieval can use the language the
    embedding model is strongest in. The original question is still what the answer is for."""
    res = _chat_create(
        [{"role": "user", "content":
          "Translate this question into English. Output only the English translation, nothing else.\n\n" + question}],
        max_tokens=400, temperature=0)
    out = (res.choices[0].message.content or "").strip()
    return out.splitlines()[0].strip() if out else ""


def generate_answer(context, question, max_tokens, stats=None, rules=""):
    """Streams Grok's answer. Reasoning deltas (if the model emits them) are wrapped in <think>…</think>
    so the UI can show them separately."""
    stream = _chat_create([{"role": "user", "content": format_prompt(context, question, rules)}],
                          max_tokens=max_tokens, temperature=TEMPERATURE, stream=True)
    in_think = False
    t0 = time.perf_counter()
    try:
        for chunk in stream:
            if getattr(chunk, "usage", None) and stats is not None:
                stats["generation_tokens"] = chunk.usage.completion_tokens
                stats["prompt_tokens"] = chunk.usage.prompt_tokens
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            d = ch.delta
            reasoning = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
            if reasoning:
                if not in_think:
                    in_think = True
                    yield "<think>"
                yield reasoning
            if d.content:
                if in_think:
                    in_think = False
                    yield "</think>"
                yield d.content
            if ch.finish_reason and stats is not None:
                stats["finish_reason"] = ch.finish_reason
    finally:
        if in_think:
            yield "</think>"
        if stats is not None:
            dt = time.perf_counter() - t0
            if stats.get("generation_tokens") and dt > 0:
                stats["generation_tps"] = stats["generation_tokens"] / dt


QUERY_STAGES = ["qembed", "mmr", "context", "infer", "answer"]


def run_query(source, question, max_tokens, answer_lang="en", boost=True):
    """One RAG turn. Yields ('stage', key, status, ms, detail), ('token', text), ('ping',) and
    finally ('done', trace). The compute lock is held for the whole turn."""
    t_all = time.perf_counter()
    v_info = lang_info(source.lang_code, source.lang_name)
    q_script = detect_script(question)
    trace = {
        "question": question, "video_id": source.video_id, "index_key": source.key, "source_id": source.id,
        "video_lang": v_info["name"], "video_lang_kind": source.lang_kind, "video_lang_reason": source.lang_reason,
        "answer_lang": answer_lang, "q_script": q_script, "v_script": v_info["script"],
        "n_chunks": source.n_chunks, "sources": [], "scores_available": False, "prompt": "",
        "bridge_query": None, "bridge_s": 0.0,
    }
    cur = "qembed"
    yield ("stage", "qembed", "running", None, "")
    try:
        load_acc = 0.0
        entry = get_index(source)
        vs, emb = entry["vs"], entry["emb"]
        t_stage = time.perf_counter()
        retrieval_query = question
        if boost and q_script in LOW_RESOURCE_SCRIPTS:
            yield ("stage", "qembed", "running", None, f"translating {q_script}-script question for retrieval")
            tb = time.perf_counter()
            bridged = bridge_query(question)
            trace["bridge_s"] = time.perf_counter() - tb
            if bridged:
                retrieval_query = bridged
                trace["bridge_query"] = bridged
        t = time.perf_counter()
        qvec = emb.embed_query(retrieval_query)
        qembed_s = time.perf_counter() - t
        stage_s = time.perf_counter() - t_stage
        extra = f" · bridged {trace['bridge_s']:.1f}s" if trace["bridge_query"] else ""
        yield ("stage", "qembed", "done", stage_s * 1000, f"dim {len(qvec)}{extra}")

        cur = "mmr"
        yield ("stage", "mmr", "running", None, "")
        t = time.perf_counter()
        scored = vs.mmr_search(qvec, k=RETRIEVE_K, fetch_k=FETCH_K, lambda_mult=LAMBDA_MULT)
        retrieved = [vs.texts[row] for row, _ in scored]
        mmr_s = time.perf_counter() - t

        sources = []
        for rank, (row, d2) in enumerate(scored, start=1):
            sources.append({"rank": rank, "chunk": row + 1, "text": vs.texts[row], "l2": d2,
                            # unit-norm vectors + squared-L2 index  =>  cosine = 1 - d2/2
                            "cos": max(-1.0, min(1.0, 1.0 - d2 / 2.0))})
        trace["sources"] = sources
        trace["scores_available"] = True
        yield ("stage", "mmr", "done", mmr_s * 1000, f"k={RETRIEVE_K} of top {FETCH_K} · λ={LAMBDA_MULT}")

        cur = "context"
        yield ("stage", "context", "running", None, "")
        t = time.perf_counter()
        context_text = "\n\n".join(retrieved)
        rules = language_rules(answer_lang, source.lang_name)
        trace["prompt"] = format_prompt(context_text, question, rules)
        ctx_s = time.perf_counter() - t
        yield ("stage", "context", "done", ctx_s * 1000, f"{len(context_text):,} chars")

        cur = "infer"
        yield ("stage", "infer", "running", None, f"{GROK_MODEL} · prefill")
        _grok_client()
        load_s = load_acc

        stats = {}
        t_gen = time.perf_counter()
        ttft = None
        pieces = []
        a_script = lang_info(answer_lang)["script"]
        eff_max = min(max_tokens * 2, 4096) if (a_script in LOW_RESOURCE_SCRIPTS or a_script == "Devanagari") else max_tokens
        trace["model"] = GROK_MODEL
        trace["max_tokens_used"] = eff_max
        for piece in generate_answer(context_text, question, eff_max, stats, rules):
            if ttft is None and piece:
                ttft = time.perf_counter() - t_gen
                yield ("stage", "infer", "running", None, "streaming")
            pieces.append(piece)
            yield ("token", piece)
        gen_s = time.perf_counter() - t_gen
        _state["llm_ok_s"] = _state["llm_ok_s"] or (ttft or gen_s)

        tokens = stats.get("generation_tokens")
        if tokens is None:
            tokens = max(1, len("".join(pieces)) // 4)
        tps = stats.get("generation_tps")
        if not tps and tokens and gen_s > 0:
            tps = tokens / gen_s
        parts = []
        if tokens is not None:
            parts.append(f"{tokens} tok")
        if tps:
            parts.append(f"{tps:.1f} tok/s")
        if ttft is not None:
            parts.append(f"TTFT {ttft:.2f}s")
        yield ("stage", "infer", "done", gen_s * 1000, " · ".join(parts))

        cur = "answer"
        total_s = time.perf_counter() - t_all
        yield ("stage", "answer", "done", total_s * 1000, "end-to-end")
        trace.update({
            "qembed_s": qembed_s, "mmr_s": mmr_s, "context_s": ctx_s, "model_load_s": load_s,
            "ttft_s": ttft, "gen_s": gen_s, "tokens": tokens, "tps": tps, "total_s": total_s,
            "prompt_tokens": stats.get("prompt_tokens"), "peak_memory_gb": stats.get("peak_memory"),
            "finish_reason": stats.get("finish_reason"), "ts": time.time(),
        })
        yield ("done", trace)
    except GeneratorExit:
        raise
    except Exception:
        yield ("stage", cur, "error", None, "failed")
        raise


# ---------------------------------------------------------------- errors / models

def friendly_error(exc):
    name, text = type(exc).__name__, str(exc)
    mapping = {
        "TranscriptsDisabled": "Captions are disabled for this video, so there is no transcript to index.",
        "NoTranscriptFound": "No transcript exists for this video in the requested language.",
        "VideoUnavailable": "This video is unavailable (removed, private, or region-locked).",
        "VideoUnplayable": "This video cannot be played, so its transcript is unavailable.",
        "InvalidVideoId": "That is not a valid YouTube video ID.",
        "AgeRestricted": "This video is age-restricted; its transcript cannot be fetched anonymously.",
        "IpBlocked": YouTubeBlocked().args[0],
        "RequestBlocked": YouTubeBlocked().args[0],
        "PoTokenRequired": YouTubeBlocked().args[0],
        "YouTubeRequestFailed": "The request to YouTube failed. Check your internet connection.",
    }
    if isinstance(exc, (LanguageUnavailable, YouTubeBlocked)):
        return text
    if name in mapping:
        return mapping[name]
    if isinstance(exc, (FileNotFoundError, ConfigError)):
        return text
    if name in ("BadRequestError", "AuthenticationError", "PermissionDeniedError", "NotFoundError"):
        msg = _provider_message(exc)
        low_msg = msg.lower()
        if "api key" in low_msg or "api_key" in low_msg or "unauthor" in low_msg:
            hint = ("xAI keys start with 'xai-' (console.x.ai); Groq keys start with 'gsk_' (console.groq.com) — "
                    "either works, set it in backend/.env and restart the backend.")
            return f"{LLM['provider']} rejected the API key: {msg} {hint}"
        if "model" in low_msg:
            return (f"{LLM['provider']} rejected the model '{GROK_MODEL}': {msg} "
                    "Set GROK_MODEL (xAI) or GROQ_MODEL (Groq) in backend/.env to a model your account can use.")
        return f"{LLM['provider']} rejected the request: {msg}"
    low = text.lower()
    if name in ("AuthenticationError", "PermissionDeniedError") or "api key not valid" in low or "api_key_invalid" in low or "401" in text[:40]:
        return "The provider rejected the API key. Check XAI_API_KEY / GEMINI_API_KEY in backend/.env and restart the backend."
    if name == "RateLimitError" or "resource_exhausted" in low or "429" in text[:60]:
        return "The provider rate limit or quota was reached. Wait a moment and retry."
    if name in ("APIConnectionError", "APITimeoutError") or "connecterror" in low:
        return "Could not reach the model provider. Check your internet connection."
    if name == "NotFoundError" or "is not found" in low or "model_not_found" in low:
        return f"The configured model was not found by the provider ({text[:120]}). Check GROK_MODEL / GEMINI_EMBED_MODEL in backend/.env."
    if isinstance(exc, ValueError):
        return text
    return f"{name}: {text}"


def dir_size_gb(path):
    total = 0
    try:
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        return None
    return total / (1024 ** 3)


@lru_cache(maxsize=8)
def model_dir_info(path):
    info = {"exists": os.path.isdir(path), "path": path, "size_gb": None, "config": {}}
    if info["exists"]:
        info["size_gb"] = dir_size_gb(path)
        cfg = os.path.join(path, "config.json")
        if os.path.isfile(cfg):
            try:
                with open(cfg, "r", encoding="utf-8") as fh:
                    info["config"] = json.load(fh)
            except Exception as exc:
                log.warning("Could not read %s: %s", cfg, exc)
    return info
