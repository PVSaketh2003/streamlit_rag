"""
MLX RAG STUDIO
==============
Local, on-device RAG workspace for Apple Silicon.

Backend (unchanged semantics):
  YouTubeTranscriptApi -> RecursiveCharacterTextSplitter(1000/200)
  -> HuggingFaceEmbeddings(Qwen3-Embedding-0.6B, normalized) -> FAISS
  -> MMR retrieval (k=3, fetch_k=15, lambda_mult=0.5)
  -> Qwen3-8B-4bit via MLX-LM (tokenizer chat template, streaming)

Everything (logic, CSS, JS, UI) lives in this single file.
"""

import gc
import html
import importlib.metadata as importlib_metadata
import json
import logging
import os
import re
import threading
import time
import uuid
from urllib.request import Request, urlopen

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

try:
    from mlx_lm import stream_generate
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("mlx_rag_studio")

# Streamlit's file watcher probes every lazily-imported `transformers` submodule and logs a
# long (harmless) traceback for optional ones that need `torchvision`. This app never uses them.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.CRITICAL)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MLX RAG Studio",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL PATHS + RETRIEVAL CONSTANTS (preserved exactly)
# ============================================================

EMBEDDING_MODEL_PATH = (
    "/Users/pvsairamsaketh/Desktop/langchain/models/"
    "qwen3-embedding-0.6b"
)
LLM_MODEL_PATH = (
    "/Users/pvsairamsaketh/Desktop/langchain/models/"
    "qwen3-8b-4bit"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVE_K = 3
FETCH_K = 15
LAMBDA_MULT = 0.5
TEMPERATURE = 0.5

DEFAULT_VIDEO = "Ilg3gGewQ5U"


# ============================================================
# PROMPT TEMPLATE — ORIGINAL RAG INSTRUCTIONS PRESERVED
# ============================================================

prompt = PromptTemplate(
    template="""You are a helpful AI assistant.
Answer the question ONLY using the provided transcript context below.
Do not assume or extrapolate any facts not directly mentioned in the text.
If the context is insufficient to answer the question, just say "I don't know."
{language_rules}
Transcript Context:
---
{context}
---

Question: {question}
Answer:""",
    input_variables=["context", "question", "language_rules"]
)


# ============================================================
# BACKEND — ORIGINAL PIPELINE (cached exactly as before)
# ============================================================

def extract_video_id(value: str) -> str:
    value = value.strip()
    patterns = [
        r"(?:youtube\.com/watch\?v=)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/watch\?.*?&v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return value


VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


@st.cache_data(show_spinner=False)
def load_transcript(video_id, lang_code="en"):
    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.fetch(video_id, languages=[lang_code])
    transcript_text = " ".join(chunk.text for chunk in transcript_list)
    return transcript_text


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={"prompt_name": "query"}
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    model, tokenizer = load(LLM_MODEL_PATH)
    return model, tokenizer


def get_retriever(vector_store):
    return vector_store.as_retriever(
        search_kwargs={"k": RETRIEVE_K, "lambda_mult": LAMBDA_MULT, "fetch_k": FETCH_K},
        search_type="mmr"
    )


def generate_answer(model, tokenizer, context, question, max_tokens, stats=None, language_rules=""):
    """Same generation as before. `stats` (optional dict) receives the real
    token counts / throughput reported by MLX-LM's last streamed response."""
    final_prompt = prompt.format(context=context, question=question, language_rules=language_rules)
    messages = [{"role": "user", "content": final_prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    sampler = make_sampler(temp=TEMPERATURE)

    if STREAMING_AVAILABLE:
        for response in stream_generate(
            model,
            tokenizer,
            prompt=chat_prompt,
            sampler=sampler,
            max_tokens=max_tokens
        ):
            if stats is not None:
                stats["generation_tokens"] = getattr(response, "generation_tokens", None)
                stats["prompt_tokens"] = getattr(response, "prompt_tokens", None)
                stats["generation_tps"] = getattr(response, "generation_tps", None)
                stats["prompt_tps"] = getattr(response, "prompt_tps", None)
                stats["peak_memory"] = getattr(response, "peak_memory", None)
                stats["finish_reason"] = getattr(response, "finish_reason", None)
            if response.text:
                yield response.text
    else:
        answer = generate(
            model,
            tokenizer,
            prompt=chat_prompt,
            sampler=sampler,
            max_tokens=max_tokens
        )
        if stats is not None:
            try:
                stats["generation_tokens"] = len(tokenizer.encode(answer))
            except Exception:
                stats["generation_tokens"] = None
        yield answer


# ============================================================
# MULTILINGUAL SUPPORT
# ============================================================
# Video language (transcript) and answer language are independent.
#  - Transcript languages: whatever caption tracks YouTube actually has for the video.
#  - Answer languages: anything Qwen3-8B can write; the list below is what the UI offers.

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
VIDEO_LANG_OPTIONS = ["auto"] + [c for c, *_ in LANGS] + ["other"]
ANSWER_LANG_OPTIONS = [c for c, *_ in LANGS]

_SCRIPT_RANGES = [
    ("Telugu", 0x0C00, 0x0C7F), ("Kannada", 0x0C80, 0x0CFF), ("Tamil", 0x0B80, 0x0BFF),
    ("Malayalam", 0x0D00, 0x0D7F), ("Bengali", 0x0980, 0x09FF), ("Gujarati", 0x0A80, 0x0AFF),
    ("Gurmukhi", 0x0A00, 0x0A7F), ("Devanagari", 0x0900, 0x097F), ("Arabic", 0x0600, 0x06FF),
    ("Arabic", 0x0750, 0x077F), ("Kana", 0x3040, 0x30FF), ("Korean", 0xAC00, 0xD7AF),
    ("Han", 0x4E00, 0x9FFF), ("Cyrillic", 0x0400, 0x04FF),
]
# Scripts where direct cross-language retrieval measured weaker with Qwen3-Embedding-0.6B
# (Telugu/Tamil measured; the rest are the same low-resource family and are treated alike).
LOW_RESOURCE_SCRIPTS = {"Telugu", "Tamil", "Kannada", "Malayalam", "Bengali", "Gujarati", "Gurmukhi", "Arabic"}


def lang_info(code, fallback_name=None):
    base = (code or "").split("-")[0].lower()
    if base in LANG_BY_CODE:
        return LANG_BY_CODE[base]
    name = fallback_name or base or "Unknown"
    return {"code": base, "name": name, "native": name, "script": EXTRA_SCRIPTS.get(base, "Latin")}


def detect_script(text):
    """Dominant Unicode script of a question (cheap, no model involved)."""
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


@st.cache_data(show_spinner=False)
def list_transcript_tracks(video_id):
    """Caption tracks YouTube really offers for this video (manual and auto-generated)."""
    transcript_list = YouTubeTranscriptApi().list(video_id)
    tracks = []
    for t in transcript_list:
        tracks.append({
            "code": t.language_code,
            "name": str(t.language).replace(" (auto-generated)", ""),
            "generated": bool(t.is_generated),
            "translatable": bool(t.is_translatable),
        })
    if not tracks:
        raise ValueError("No caption tracks were found for this video.")
    return tracks


def pick_track(tracks, requested):
    """Choose a real caption track. Manual captions win over auto-generated ones
    for the same language (this mirrors what YouTubeTranscriptApi.fetch does)."""
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
    """Answer-language instructions appended to the original grounding prompt."""
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


def bridge_query(model, tokenizer, question):
    """Translate a short question to English with the already-loaded Qwen3-8B so retrieval
    can use the language the embedding model is strongest in. The original question is
    still what the answer is generated for."""
    msgs = [{"role": "user", "content":
             "Translate this question into English. Output only the English translation, nothing else.\n\n" + question}]
    try:
        chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    out = generate(model, tokenizer, prompt=chat, max_tokens=120).strip()
    if "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    return out.splitlines()[0].strip() if out else ""


# ============================================================
# STUDIO STATE — process-wide store (survives reruns + refresh)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_store():
    return {
        "convs": {},        # conv_id -> conversation dict
        "indexes": {},      # video_id -> index record
        "perf": [],         # one record per answered query
        "building": set(),  # video ids currently being indexed
        "transcripts_seen": set(),
        "lock": threading.Lock(),
    }


class TimedEmbeddings(Embeddings):
    """Thin pass-through that measures time spent inside the real
    embedding model. FAISS.from_documents is still called unchanged."""

    def __init__(self, inner):
        self.inner = inner
        self.doc_s = 0.0
        self.query_s = 0.0

    def embed_documents(self, texts):
        t = time.perf_counter()
        out = self.inner.embed_documents(texts)
        self.doc_s += time.perf_counter() - t
        return out

    def embed_query(self, text):
        t = time.perf_counter()
        out = self.inner.embed_query(text)
        self.query_s += time.perf_counter() - t
        return out


# ---------- pipeline stage definitions (shown in the inspector) ----------

STAGES = [
    ("fetch",   "Transcript fetch",       "idx"),
    ("split",   "Text splitting",         "idx"),
    ("embed",   "Embedding generation",   "idx"),
    ("faiss",   "FAISS indexing",         "idx"),
    ("qembed",  "Query embedding",        "qry"),
    ("mmr",     "MMR retrieval",          "qry"),
    ("context", "Context construction",   "qry"),
    ("infer",   "MLX inference",          "qry"),
    ("answer",  "Final answer",           "qry"),
]
QUERY_KEYS = [k for k, _, kind in STAGES if kind == "qry"]


def empty_stage_state():
    return {k: {"status": "idle", "ms": None, "detail": ""} for k in QUERY_KEYS}


def get_embeddings_timed():
    store = get_store()
    if not os.path.isdir(EMBEDDING_MODEL_PATH):
        raise FileNotFoundError(f"Embedding model directory not found: {EMBEDDING_MODEL_PATH}")
    first = "embed_load_s" not in store
    t = time.perf_counter()
    emb = load_embeddings()
    dt = time.perf_counter() - t
    if first:
        store["embed_load_s"] = dt
    return emb, (dt if first else 0.0)


def get_llm_timed():
    store = get_store()
    if not os.path.isdir(LLM_MODEL_PATH):
        raise FileNotFoundError(f"LLM directory not found: {LLM_MODEL_PATH}")
    first = "llm_load_s" not in store
    t = time.perf_counter()
    model, tokenizer = load_llm()
    dt = time.perf_counter() - t
    if first:
        store["llm_load_s"] = dt
    return model, tokenizer, (dt if first else 0.0)


def unload_llm():
    store = get_store()
    load_llm.clear()
    store.pop("llm_load_s", None)
    gc.collect()
    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception as exc:
        log.warning("MLX cache clear failed: %s", exc)


@st.cache_data(show_spinner=False)
def fetch_video_title(video_id):
    """Best-effort title via YouTube oEmbed (video id only is sent)."""
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = Request(url, headers={"User-Agent": "mlx-rag-studio"})
        with urlopen(req, timeout=4) as resp:
            return json.loads(resp.read().decode("utf-8")).get("title")
    except Exception as exc:
        log.info("Title lookup skipped for %s (%s)", video_id, type(exc).__name__)
        return None


def build_index(video_id, track, reason, tracks):
    """Index one video in one caption language. Yields ('stage', key, status, ms, detail)
    events and finally ('done', record). Pipeline logic is identical to the original:
    transcript -> split -> embed -> FAISS."""
    store = get_store()
    code = track["code"]
    key = index_key(video_id, code)
    info = lang_info(code, track["name"])
    timings, details = {}, {}
    cur = "fetch"
    try:
        yield ("stage", "fetch", "running", None, f"{info['name']} captions")
        t = time.perf_counter()
        was_cached = key in store["transcripts_seen"]
        transcript_text = load_transcript(video_id, code)
        timings["fetch"] = time.perf_counter() - t
        store["transcripts_seen"].add(key)
        if not transcript_text.strip():
            raise ValueError("The transcript is empty.")
        details["fetch"] = (f"{len(transcript_text):,} chars · {info['name']} · "
                            + ("from cache" if was_cached else "downloaded"))
        yield ("stage", "fetch", "done", timings["fetch"] * 1000, details["fetch"])

        cur = "split"
        yield ("stage", "split", "running", None, "")
        t = time.perf_counter()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.create_documents([transcript_text])
        timings["split"] = time.perf_counter() - t
        if not chunks:
            raise ValueError("Splitting produced no chunks.")
        details["split"] = f"{len(chunks):,} chunks · {CHUNK_SIZE}/{CHUNK_OVERLAP}"
        yield ("stage", "split", "done", timings["split"] * 1000, details["split"])

        cur = "embed"
        yield ("stage", "embed", "running", None, "loading embedding model" if "embed_load_s" not in store else "")
        base_emb, embed_load_s = get_embeddings_timed()
        emb = TimedEmbeddings(base_emb)
        t = time.perf_counter()
        vector_store = FAISS.from_documents(chunks, emb)
        total_build = time.perf_counter() - t
        timings["embed"] = emb.doc_s
        timings["faiss"] = max(total_build - emb.doc_s, 0.0)
        details["embed"] = f"{len(chunks):,} vectors" + (f" · model load {embed_load_s:.2f}s" if embed_load_s else "")
        details["faiss"] = f"{vector_store.index.ntotal:,} vectors · dim {vector_store.index.d}"
        yield ("stage", "embed", "done", timings["embed"] * 1000, details["embed"])
        cur = "faiss"
        yield ("stage", "faiss", "done", timings["faiss"] * 1000, details["faiss"])

        content_idx = {}
        for i, c in enumerate(chunks):
            content_idx.setdefault(c.page_content, i)

        record = {
            "key": key,
            "video_id": video_id,
            "title": fetch_video_title(video_id),
            "lang_code": code,
            "lang_name": info["name"],
            "lang_kind": "auto-generated" if track["generated"] else "manual",
            "lang_reason": reason,
            "tracks": tracks,
            "vs": vector_store,
            "emb": emb,
            "content_idx": content_idx,
            "n_chunks": len(chunks),
            "n_chars": len(transcript_text),
            "n_words": len(transcript_text.split()),
            "timings": timings,
            "details": details,
            "total_s": sum(timings.values()),
            "built_at": time.time(),
        }
        yield ("done", record)
    except Exception as exc:
        yield ("stage", cur, "error", None, str(exc)[:160])
        raise


def run_query(rec, question, max_tokens, answer_lang="en", boost=True):
    """One RAG turn. Yields ('stage', key, status, ms, detail),
    ('token', text) and finally ('done', trace)."""
    store = get_store()
    vs, emb = rec["vs"], rec["emb"]
    t_all = time.perf_counter()
    v_info = lang_info(rec["lang_code"], rec["lang_name"])
    q_script = detect_script(question)
    trace = {
        "question": question,
        "video_id": rec["video_id"],
        "index_key": rec["key"],
        "video_lang": v_info["name"],
        "answer_lang": answer_lang,
        "q_script": q_script,
        "v_script": v_info["script"],
        "stages": empty_stage_state(),
        "sources": [],
        "scores_available": False,
        "prompt": "",
        "bridge_query": None,
        "bridge_s": 0.0,
    }
    cur = "qembed"
    load_acc = 0.0
    try:
        # --- retrieval: identical MMR to the original retriever ---
        yield ("stage", "qembed", "running", None, "")
        t_stage = time.perf_counter()
        retrieval_query = question
        if boost and q_script in LOW_RESOURCE_SCRIPTS:
            yield ("stage", "qembed", "running", None, f"translating {q_script}-script question for retrieval")
            model_b, tok_b, load_b = get_llm_timed()
            load_acc += load_b
            tb = time.perf_counter()
            bridged = bridge_query(model_b, tok_b, question)
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
        scored = None
        try:
            scored = vs.max_marginal_relevance_search_with_score_by_vector(
                qvec, k=RETRIEVE_K, fetch_k=FETCH_K, lambda_mult=LAMBDA_MULT
            )
            retrieved_docs = [d for d, _ in scored]
        except Exception as exc:  # fall back to the original retriever path
            log.warning("Scored MMR unavailable, using retriever: %s", exc)
            scored = None
            retrieved_docs = get_retriever(vs).invoke(retrieval_query)
        mmr_s = time.perf_counter() - t

        sources = []
        for rank, doc in enumerate(retrieved_docs, start=1):
            entry = {
                "rank": rank,
                "chunk": rec["content_idx"].get(doc.page_content, -1) + 1,
                "text": doc.page_content,
                "l2": None,
                "cos": None,
            }
            if scored is not None:
                d2 = float(scored[rank - 1][1])
                entry["l2"] = d2
                # FAISS IndexFlatL2 returns squared L2; embeddings are unit-norm,
                # so cosine similarity = 1 - d2 / 2.
                entry["cos"] = max(-1.0, min(1.0, 1.0 - d2 / 2.0))
            sources.append(entry)
        trace["sources"] = sources
        trace["scores_available"] = scored is not None
        detail = f"k={RETRIEVE_K} of top {FETCH_K} · λ={LAMBDA_MULT}"
        yield ("stage", "mmr", "done", mmr_s * 1000, detail)

        cur = "context"
        yield ("stage", "context", "running", None, "")
        t = time.perf_counter()
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        rules = language_rules(answer_lang, rec["lang_name"])
        trace["prompt"] = prompt.format(context=context_text, question=question, language_rules=rules)
        ctx_s = time.perf_counter() - t
        yield ("stage", "context", "done", ctx_s * 1000, f"{len(context_text):,} chars")

        # --- inference ---
        cur = "infer"
        first_load = "llm_load_s" not in store
        yield ("stage", "infer", "running", None, "loading Qwen3-8B (first use)" if first_load else "prefill")
        model, tokenizer, load_s = get_llm_timed()
        load_s += load_acc

        stats = {}
        t_gen = time.perf_counter()
        ttft = None
        pieces = []
        # Indic and Arabic-script text needs many more tokens per word, so give those answers more room.
        a_script = lang_info(answer_lang)["script"]
        eff_max = min(max_tokens * 2, 4096) if (a_script in LOW_RESOURCE_SCRIPTS or a_script == "Devanagari") else max_tokens
        trace["max_tokens_used"] = eff_max
        for piece in generate_answer(model, tokenizer, context_text, question, eff_max, stats, rules):
            if ttft is None and piece:
                ttft = time.perf_counter() - t_gen
                yield ("stage", "infer", "running", None, "streaming")
            pieces.append(piece)
            yield ("token", piece)
        gen_s = time.perf_counter() - t_gen

        tokens = stats.get("generation_tokens")
        if tokens is None:
            try:
                tokens = len(tokenizer.encode("".join(pieces)))
            except Exception:
                tokens = None
        tps = stats.get("generation_tps")
        if not tps and tokens and gen_s > 0:
            tps = tokens / gen_s
        peak = stats.get("peak_memory")

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
            "qembed_s": qembed_s, "mmr_s": mmr_s, "context_s": ctx_s,
            "model_load_s": load_s, "ttft_s": ttft, "gen_s": gen_s,
            "tokens": tokens, "tps": tps, "total_s": total_s,
            "prompt_tokens": stats.get("prompt_tokens"),
            "peak_memory_gb": peak,
            "finish_reason": stats.get("finish_reason"),
            "ts": time.time(),
        })
        store["perf"].append({
            "ts": trace["ts"], "video_id": rec["video_id"],
            "retrieval_s": trace["bridge_s"] + qembed_s + mmr_s, "model_load_s": load_s,
            "ttft_s": ttft, "gen_s": gen_s, "tokens": tokens, "tps": tps,
            "total_s": total_s,
        })
        yield ("done", trace)
    except Exception as exc:
        yield ("stage", cur, "error", None, str(exc)[:160])
        raise


# ============================================================
# ERROR TRANSLATION
# ============================================================

def friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    text = str(exc)
    mapping = {
        "TranscriptsDisabled": "Captions are disabled for this video, so there is no transcript to index.",
        "NoTranscriptFound": "No Hindi or English transcript exists for this video.",
        "VideoUnavailable": "This video is unavailable (removed, private, or region-locked).",
        "VideoUnplayable": "This video cannot be played, so its transcript is unavailable.",
        "InvalidVideoId": "That is not a valid YouTube video ID.",
        "AgeRestricted": "This video is age-restricted; its transcript cannot be fetched anonymously.",
        "IpBlocked": "YouTube is blocking transcript requests from this IP right now. Try again later.",
        "RequestBlocked": "YouTube blocked the transcript request. Try again later or from another network.",
        "PoTokenRequired": "YouTube requires a proof-of-origin token for this video's transcript.",
        "YouTubeRequestFailed": "The request to YouTube failed. Check your internet connection.",
    }
    if isinstance(exc, LanguageUnavailable):
        return text
    if name in mapping:
        return mapping[name]
    if isinstance(exc, FileNotFoundError):
        return text
    if isinstance(exc, MemoryError) or "memory" in text.lower() or "out of memory" in text.lower():
        return ("The system ran low on memory. Unload the model in Models, close other heavy apps, "
                "or lower the maximum response length, then retry.")
    if isinstance(exc, ValueError):
        return text
    return f"{name}: {text}"


# ============================================================
# SMALL FORMATTERS + HTML BUILDERS
# ============================================================

esc = html.escape


def fmt_s(s):
    if s is None:
        return "—"
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s"


def fmt_ms(ms):
    return "—" if ms is None else fmt_s(ms / 1000.0)


def fmt_num(n):
    return "—" if n is None else f"{n:,}"


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


@st.cache_data(show_spinner=False)
def model_dir_info(path):
    info = {"exists": os.path.isdir(path), "size_gb": None, "config": {}}
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


def split_reasoning(raw: str):
    """Separate Qwen3 <think> reasoning from the final answer (display only)."""
    if "</think>" in raw:
        before, after = raw.split("</think>", 1)
        return before.replace("<think>", "").strip(), after.strip(), False
    if raw.lstrip().startswith("<think>"):
        return raw.lstrip()[len("<think>"):].strip(), "", True
    return "", raw, False


def stat_tile(label, value, sub="", tone=""):
    return (
        f'<div class="tile {tone}"><div class="tile-l">{esc(label)}</div>'
        f'<div class="tile-v">{value}</div><div class="tile-s">{sub}</div></div>'
    )


def chip(text, kind=""):
    return f'<span class="chip {kind}">{text}</span>'


SEG_COLORS = ["#315CEB", "#087E8B", "#7C8FD0", "#16845D", "#B7791F", "#94A3B8"]


def stacked_bar_html(segments):
    """segments: list of (label, seconds). Widths are proportional to real values."""
    segs = [(l, v) for l, v in segments if v and v > 0]
    total = sum(v for _, v in segs)
    if total <= 0:
        return '<div class="empty-note">No measurements yet.</div>'
    bar, legend = [], []
    for i, (label, v) in enumerate(segs):
        color = SEG_COLORS[i % len(SEG_COLORS)]
        bar.append(
            f'<i style="width:{v / total * 100:.2f}%;background:{color}" '
            f'title="{esc(label)}: {fmt_s(v)}"></i>'
        )
        legend.append(
            f'<span class="lg"><b style="background:{color}"></b>{esc(label)}'
            f'<em>{fmt_s(v)}</em></span>'
        )
    return (
        f'<div class="stack">{"".join(bar)}</div>'
        f'<div class="legend">{"".join(legend)}</div>'
        f'<div class="stack-total">Total {fmt_s(total)}</div>'
    )


def transcript_map_html(n_chunks, hits):
    """hits: list of (rank, chunk_number_1_based). Position map of retrieved
    chunks across the transcript (chunks are sequential in the transcript)."""
    if not n_chunks:
        return ""
    cells = min(n_chunks, 120)
    marks = {}
    for rank, chunk in hits:
        if chunk > 0:
            cell = min(cells - 1, (chunk - 1) * cells // n_chunks)
            marks.setdefault(cell, []).append(rank)
    out = []
    for c in range(cells):
        if c in marks:
            r = min(marks[c])
            out.append(f'<i class="hit" title="retrieved #{r}"><u>{r}</u></i>')
        else:
            out.append("<i></i>")
    return (
        f'<div class="tmap">{"".join(out)}</div>'
        f'<div class="tmap-axis"><span>start</span><span>transcript position · {n_chunks} chunks</span>'
        f'<span>end</span></div>'
    )


def chunk_cards_html(sources, scores_available):
    if not sources:
        return '<div class="empty-note">No chunks retrieved for this answer yet.</div>'
    cards = []
    for s in sources:
        score_html = ""
        if s.get("cos") is not None:
            pct = max(0.0, min(1.0, s["cos"])) * 100
            score_html = (
                f'<div class="score"><span>cosine similarity</span><b>{s["cos"]:.3f}</b></div>'
                f'<div class="score-bar"><i style="width:{pct:.1f}%"></i></div>'
            )
        chunk_label = f"chunk #{s['chunk']}" if s["chunk"] > 0 else "chunk"
        cards.append(
            f'<div class="chunk"><div class="chunk-h"><span class="rank">{s["rank"]}</span>'
            f'<span class="chunk-id">{chunk_label}</span>'
            f'<span class="chunk-ord">retrieval order {s["rank"]}</span></div>'
            f'{score_html}<div class="chunk-t">{esc(s["text"])}</div></div>'
        )
    note = ""
    if not scores_available:
        note = '<div class="empty-note">Scores unavailable for this retrieval path.</div>'
    return note + "".join(cards)


# ============================================================
# DESIGN SYSTEM (light theme only) — CSS
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --bg:#F4F6FA; --surface:#FFFFFF; --text:#172033; --muted:#64748B;
  --accent:#315CEB; --accent2:#087E8B; --border:#E3E8F0; --ok:#16845D;
  --tint:#EEF2FF; --soft:#F8FAFC; --warn:#B7791F; --err:#B42318;
  --r:10px; --r-sm:7px;
  --sh:0 1px 2px rgba(23,32,51,.05), 0 1px 1px rgba(23,32,51,.03);
  --sh-lg:0 8px 24px rgba(23,32,51,.08);
  --ui:'Inter',-apple-system,BlinkMacSystemFont,'SF Pro Text','Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,'SF Mono',Menlo,monospace;
  color-scheme: light;
}

/* ---------- base canvas ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
  background: var(--bg) !important; color: var(--text) !important; color-scheme: light;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp h5, .stApp h6, .stApp button, .stApp input, .stApp textarea, .stApp td, .stApp th,
.stApp [data-baseweb] { font-family: var(--ui) !important; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] { color: var(--text); }
.stApp code, .stApp pre, .mono { font-family: var(--mono) !important; }
[data-testid="stHeader"] { background: transparent !important; height: 2.4rem; }
[data-testid="stDecoration"] { display:none !important; }
[data-testid="stToolbar"] { right: .6rem; }
[data-testid="stToolbar"] * { color: var(--muted) !important; }
.block-container { max-width: 1500px !important; padding: 1.1rem 2rem 6.5rem !important; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
hr { border-color: var(--border) !important; }
::selection { background: #DCE5FF; }
a { color: var(--accent) !important; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] > div:first-child { background: var(--surface) !important; }
[data-testid="stSidebarContent"] { padding: 0 .2rem; }
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { background: transparent; }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding: .9rem .8rem 1rem !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .25rem; }

.brand { display:flex; align-items:center; gap:10px; padding:2px 4px 10px; }
.brand-mark { width:30px; height:30px; border-radius:8px; background:var(--text); color:#fff;
  display:flex; align-items:center; justify-content:center; font:600 12px var(--mono); letter-spacing:-.5px;
  position:relative; }
.brand-mark::after { content:''; position:absolute; right:-3px; bottom:-3px; width:9px; height:9px;
  border-radius:50%; background:var(--accent); border:2px solid #fff; }
.brand-name { color:var(--text) !important; font-weight:650; font-size:14.5px; letter-spacing:-.2px; line-height:1.1; }
.brand-sub { font:500 10.5px var(--mono); color:var(--muted); letter-spacing:.3px; }

.side-label { font:600 10.5px var(--mono); letter-spacing:1.1px; color:var(--muted);
  text-transform:uppercase; padding:14px 6px 4px; }

/* nav buttons */
[class*="st-key-nav_"] button, [class*="st-key-navon_"] button,
[class*="st-key-conv_"] button, [class*="st-key-convon_"] button {
  justify-content:flex-start !important; text-align:left; background:transparent !important;
  border:1px solid transparent !important; color:var(--text) !important; font-weight:500 !important;
  min-height:34px; padding:.25rem .6rem !important; border-radius:var(--r-sm) !important;
  box-shadow:none !important; font-size:13.5px !important;
}
[class*="st-key-nav_"] button > div, [class*="st-key-navon_"] button > div,
[class*="st-key-conv_"] button > div, [class*="st-key-convon_"] button > div { justify-content:flex-start !important; width:100%; text-align:left !important; }
[class*="st-key-nav_"] button p, [class*="st-key-navon_"] button p { text-align:left !important; }
[data-testid="stAppDeployButton"], .stDeployButton { display:none !important; }
[class*="st-key-nav_"] button:hover, [class*="st-key-conv_"] button:hover { background:var(--soft) !important; border-color:var(--border) !important; }
[class*="st-key-navon_"] button, [class*="st-key-convon_"] button {
  background:var(--tint) !important; color:var(--accent) !important; border-color:#D7E0FF !important; font-weight:600 !important; }
[class*="st-key-navon_"] button *, [class*="st-key-convon_"] button * { color:var(--accent) !important; }
[class*="st-key-conv_"] button p, [class*="st-key-convon_"] button p { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
[class*="st-key-conv_"] button, [class*="st-key-convon_"] button { font-size:13px !important; font-weight:450 !important; }

.st-key-newchat button { background:var(--text) !important; color:#fff !important; border:0 !important;
  min-height:38px; border-radius:var(--r-sm) !important; box-shadow:var(--sh); font-weight:550 !important; }
.st-key-newchat button * { color:#fff !important; }
.st-key-newchat button:hover { background:#27324D !important; }

.side-status { font:500 11.5px var(--mono); color:var(--muted); display:flex; flex-direction:column; gap:6px;
  padding:10px 8px; border:1px solid var(--border); border-radius:var(--r); background:var(--soft); }
.side-status .row { display:flex; justify-content:space-between; align-items:center; gap:6px; }
.side-status .row span:first-child { color:var(--text); }
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#CBD5E1; margin-right:6px; }
.dot.on { background:var(--ok); }
.dot.run { background:var(--accent); animation:pulse 1.2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ---------- buttons ---------- */
.stButton > button, [data-testid="stPopover"] > button, [data-testid^="stBaseButton"] {
  font-family: var(--ui) !important; border-radius:var(--r-sm) !important; font-weight:500 !important;
  font-size:13px !important; transition: background .12s, border-color .12s, box-shadow .12s; }
[data-testid="stBaseButton-secondary"], [data-testid="stPopoverButton"] {
  background:var(--surface) !important; color:var(--text) !important; border:1px solid var(--border) !important;
  box-shadow:var(--sh) !important; }
[data-testid="stBaseButton-secondary"] *, [data-testid="stPopoverButton"] * { color:var(--text) !important; }
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stPopoverButton"]:hover {
  border-color:#C9D3E6 !important; background:var(--soft) !important; }
[data-testid="stBaseButton-primary"] { background:var(--accent) !important; border:1px solid var(--accent) !important;
  box-shadow:0 1px 2px rgba(49,92,235,.35) !important; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] * { color:#fff !important; }
[data-testid="stBaseButton-primary"]:hover { background:#2A50D2 !important; }
[data-testid="stBaseButton-tertiary"] { background:transparent !important; color:var(--muted) !important; border:0 !important; }
[data-testid="stBaseButton-tertiary"] * { color:var(--muted) !important; }
button:disabled, button[disabled] { opacity:.5 !important; }
button:focus-visible, input:focus-visible, textarea:focus-visible { outline:2px solid rgba(49,92,235,.45) !important; outline-offset:1px; }

/* ---------- inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {
  background:var(--surface) !important; border-color:var(--border) !important; border-radius:var(--r-sm) !important; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input, textarea, [data-baseweb="select"] * {
  color:var(--text) !important; -webkit-text-fill-color:var(--text) !important; background:transparent !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(49,92,235,.12) !important; }
[data-testid="stTextInputRootElement"], [data-testid="stTextInputRootElement"] > div, [data-baseweb="input"] > div,
[data-testid="stTextArea"] [data-baseweb="textarea"], [data-testid="stSelectbox"] [data-baseweb="select"] > div { background:var(--surface) !important; background-color:var(--surface) !important; border-color:var(--border) !important; }
[data-testid="stTextInputRootElement"] { border:1px solid var(--border) !important; border-radius:var(--r-sm) !important; }
[data-testid="stSelectbox"] [data-baseweb="select"], [data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div { background:var(--surface) !important; background-color:var(--surface) !important; }
[data-testid="stSelectbox"] svg { fill:var(--muted) !important; }
[data-testid="stSelectbox"] div, [data-testid="stTextInput"] div[data-baseweb] , [data-testid="stTextInput"] div[data-baseweb] > div { background-color:var(--surface) !important; }
[data-testid="stSelectbox"] div, [data-testid="stSelectbox"] span, [data-testid="stSelectbox"] input { color:var(--text) !important; -webkit-text-fill-color:var(--text) !important; }
label:has(input[role="switch"]) > div:first-of-type { background:#CBD5E1 !important; background-color:#CBD5E1 !important; }
label:has(input[role="switch"]:checked) > div:first-of-type { background:var(--accent) !important; background-color:var(--accent) !important; }
label:has(input[role="switch"]) > div:first-of-type > div { background:#fff !important; background-color:#fff !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input { color:var(--text) !important; -webkit-text-fill-color:var(--text) !important; caret-color:var(--accent); }
details, details > summary, [data-testid="stExpander"] details, [data-testid="stExpanderDetails"], [data-testid="stStatus"] details, [data-testid="stStatus"] summary { background:var(--surface) !important; background-color:var(--surface) !important; color:var(--text) !important; }
details > summary * { color:var(--text) !important; }
input::placeholder, textarea::placeholder { color:#94A3B8 !important; -webkit-text-fill-color:#94A3B8 !important; }
[data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] { background:var(--surface) !important;
  border:1px solid var(--border) !important; border-radius:var(--r) !important; box-shadow:var(--sh-lg) !important; }
li[role="option"], [data-baseweb="menu"] li { color:var(--text) !important; background:transparent !important; }
li[role="option"]:hover, li[aria-selected="true"] { background:var(--tint) !important; }
[data-testid="stPopoverBody"] { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:var(--r) !important; }
[data-testid="stTooltipContent"] { background:var(--text) !important; color:#fff !important; }
[data-testid="stTooltipContent"] * { color:#fff !important; }
label[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p { color:var(--text) !important; font-weight:500; font-size:13px; }
[data-baseweb="checkbox"] > div:first-child { background:#CBD5E1 !important; }
label[data-baseweb="checkbox"]:has(input:checked) > div:first-child { background:var(--accent) !important; }
[data-baseweb="tab-highlight"] { background-color:var(--accent) !important; height:2px !important; }
[data-baseweb="tab-list"] { gap:6px; }
[role="tablist"] ~ div:not([role="tabpanel"]), [role="tablist"] > div[aria-hidden] { background-color:var(--accent) !important; background:var(--accent) !important; }
[data-testid="stTabs"] div[style*="255, 75, 75"], [data-testid="stTabs"] div[style*="#ff4b4b"] { background-color:var(--accent) !important; }
.react-aria-SelectionIndicator { background:var(--accent) !important; background-color:var(--accent) !important; }
[data-testid="stChatInput"] textarea:focus, [data-testid="stChatInput"] textarea:focus-visible { outline:none !important; box-shadow:none !important; border:0 !important; }
[role="tab"] { color:var(--muted) !important; } [role="tab"][aria-selected="true"], [role="tab"][aria-selected="true"] * { color:var(--accent) !important; }
[data-testid="stChatInputSubmitButton"]:disabled { background:#E3E8F0 !important; }
[data-testid="stChatInputSubmitButton"]:disabled * { color:#94A3B8 !important; fill:#94A3B8 !important; }
[data-testid="stSlider"] [role="slider"] { background:var(--accent) !important; box-shadow:none !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stSlider"] div { color:var(--text) !important; }

/* ---------- chat input ---------- */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background:transparent !important; }
[data-testid="stBottom"]::before { content:''; position:absolute; inset:-28px 0 0 0; background:linear-gradient(180deg, rgba(244,246,250,0), var(--bg) 40%); pointer-events:none; }
[data-testid="stChatInput"] { background:var(--surface) !important; border:1px solid var(--border) !important;
  border-radius:14px !important; box-shadow:var(--sh-lg) !important; max-width:860px; margin:0 auto; }
[data-testid="stChatInput"]:focus-within { border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(49,92,235,.12), var(--sh-lg) !important; }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-baseweb] { background:transparent !important; border:0 !important; box-shadow:none !important; }
[data-testid="stChatInput"] textarea { color:var(--text) !important; -webkit-text-fill-color:var(--text) !important; font-size:14.5px !important; }
[data-testid="stChatInputSubmitButton"] { background:var(--accent) !important; border-radius:9px !important; }
[data-testid="stChatInputSubmitButton"] * { color:#fff !important; fill:#fff !important; }
[data-testid="stChatInputSubmitButton"]:disabled { background:#CBD5E1 !important; }

/* ---------- chat messages ---------- */
[data-testid="stChatMessage"] { background:transparent !important; border:0 !important; padding:.55rem 0 !important; gap:.8rem; max-width:860px; margin:0 auto; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:var(--tint) !important;
  border:1px solid #DCE5FF !important; border-radius:12px !important; padding:.65rem .9rem !important; margin:.5rem auto .3rem !important; }
[data-testid="stChatMessageAvatarUser"] { background:var(--text) !important; color:#fff !important; }
[data-testid="stChatMessageAvatarUser"] * { color:#fff !important; }
[data-testid="stChatMessageAvatarAssistant"] { background:var(--surface) !important; border:1px solid var(--border); color:var(--accent) !important; }
[data-testid="stChatMessageAvatarAssistant"] * { color:var(--accent) !important; }
[data-testid="stChatMessageContent"] p { font-size:15px; line-height:1.68; }
[data-testid="stChatMessageContent"] li { line-height:1.65; }
.stApp :not(pre) > code { background:var(--tint) !important; color:#2447C7 !important; border-radius:5px; padding:.12em .38em; font-size:.88em; }
[data-testid="stCode"], [data-testid="stCode"] pre, .stApp pre { background:#F8FAFC !important; border:1px solid var(--border) !important; border-radius:var(--r) !important; }
.stApp pre code, [data-testid="stCode"] code { background:transparent !important; color:var(--text) !important; }
[data-testid="stCode"] span { color:inherit; }
[data-testid="stMarkdownContainer"] table { border-collapse:collapse; }
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td { border:1px solid var(--border); padding:6px 10px; }
[data-testid="stMarkdownContainer"] th { background:var(--soft); }
[data-testid="stMarkdownContainer"] blockquote { border-left:3px solid var(--accent); color:var(--muted); }

/* ---------- containers ---------- */
[data-testid="stExpander"] { background:var(--surface) !important; border:1px solid var(--border) !important; border-radius:var(--r) !important; box-shadow:none !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:var(--text) !important; }
[data-testid="stExpander"] details { border:0 !important; }
[data-testid="stStatus"], [data-testid="stStatusWidget"] { background:var(--surface) !important; border-color:var(--border) !important; color:var(--text) !important; }
[data-testid="stAlert"] { border-radius:var(--r) !important; border:1px solid var(--border) !important; }
[data-testid="stAlert"] * { color:var(--text) !important; }
[data-testid="stAlertContainer"] { background:var(--surface) !important; }
[data-testid="stTabs"] [role="tablist"] { gap:2px; border-bottom:1px solid var(--border) !important; }
[data-testid="stTabs"] button[role="tab"] { color:var(--muted) !important; font-size:12.5px !important; font-weight:550 !important; padding:.35rem .7rem; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:var(--accent) !important; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--accent) !important; height:2px; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background:transparent !important; }
[data-testid="stDialog"] > div, div[role="dialog"] { background:var(--surface) !important; color:var(--text) !important; }
[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border); border-radius:var(--r); padding:12px 14px; }
[data-testid="stSpinner"] * { color:var(--muted) !important; }

/* ---------- header ---------- */
.hdr { display:flex; align-items:center; gap:10px; min-height:38px; }
.crumb-src { display:inline-flex; align-items:center; gap:8px; padding:4px 10px 4px 5px; background:var(--surface);
  border:1px solid var(--border); border-radius:999px; font:500 12px var(--mono); color:var(--text); box-shadow:var(--sh); max-width:340px; }
.crumb-src img { width:26px; height:26px; border-radius:50%; object-fit:cover; background:var(--tint); }
.crumb-src .ph { width:26px; height:26px; border-radius:50%; background:var(--tint); display:inline-block; }
.crumb-src span.t { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.crumb-src.none { padding:5px 12px; color:var(--muted); }
.crumb-sep { color:#B6C0D2; font-size:16px; }
.conv-title { font-weight:620; font-size:17px; letter-spacing:-.25px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.view-title { font-weight:650; font-size:24px; letter-spacing:-.5px; margin:2px 0 2px; }
.view-sub { color:var(--muted); font-size:13.5px; margin-bottom:12px; }
.rule { height:1px; background:var(--border); margin:10px 0 6px; }

/* ---------- chips / tiles ---------- */
.chip { display:inline-flex; align-items:center; gap:5px; padding:2px 8px; border-radius:999px; font:500 11px var(--mono);
  border:1px solid var(--border); background:var(--soft); color:var(--muted); white-space:nowrap; }
.chip.ok { color:var(--ok); background:#ECFBF4; border-color:#C5EBDA; }
.chip.acc { color:var(--accent); background:var(--tint); border-color:#D7E0FF; }
.chip.warn { color:var(--warn); background:#FFF8EA; border-color:#F2E2BC; }
.chip.err { color:var(--err); background:#FEF3F2; border-color:#FAD4D0; }
.meta { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:8px 0 2px; font:500 11.5px var(--mono); color:var(--muted); }
.meta .sep { color:#CBD5E1; }
.srcs { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; align-items:center; }
.srcs .lbl { font:600 10.5px var(--mono); letter-spacing:.9px; color:var(--muted); text-transform:uppercase; margin-right:2px; }

.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; margin:6px 0 14px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:var(--r); padding:12px 14px; box-shadow:var(--sh); }
.tile-l { font:600 10.5px var(--mono); letter-spacing:.9px; text-transform:uppercase; color:var(--muted); }
.tile-v { font:600 22px var(--mono); letter-spacing:-.8px; margin-top:6px; color:var(--text); }
.tile-s { font-size:11.5px; color:var(--muted); margin-top:2px; min-height:15px; }
.tile.acc .tile-v { color:var(--accent); }
.tile.ok .tile-v { color:var(--ok); }

.panel { background:var(--surface); border:1px solid var(--border); border-radius:var(--r); padding:16px 18px; box-shadow:var(--sh); margin-bottom:12px; }
.panel h4 { margin:0 0 2px; font-size:13.5px; font-weight:620; letter-spacing:-.1px; }
.panel .sub { color:var(--muted); font-size:12px; margin-bottom:12px; }
.sec { font:600 10.5px var(--mono); letter-spacing:1.1px; text-transform:uppercase; color:var(--muted); margin:18px 0 8px; }
.kv { margin:0; padding:0; display:grid; grid-template-columns:150px 1fr; gap:6px 14px; font-size:12.5px; }
.kv dt { margin:0; padding:0; color:var(--muted); } .kv dd { margin:0; font-family:var(--mono); font-size:12px; word-break:break-all; }
.empty-note { color:var(--muted); font-size:12.5px; padding:10px 0; }

/* ---------- stacked bar ---------- */
.stack { display:flex; height:14px; border-radius:7px; overflow:hidden; background:#EDF1F7; gap:2px; }
.stack i { display:block; min-width:3px; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; margin-top:10px; }
.lg { font-size:12px; color:var(--text); display:inline-flex; align-items:center; gap:6px; }
.lg b { width:9px; height:9px; border-radius:2px; display:inline-block; }
.lg em { font:500 11.5px var(--mono); color:var(--muted); font-style:normal; }
.stack-total { font:500 11.5px var(--mono); color:var(--muted); margin-top:8px; }

/* ---------- inspector ---------- */
.st-key-inspector { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:14px 14px 10px; box-shadow:var(--sh);
  position:sticky; top:2.6rem; max-height:calc(100vh - 3.4rem); overflow-y:auto; }
.insp-h { display:flex; align-items:center; justify-content:space-between; margin-bottom:2px; }
.insp-t { font-weight:650; font-size:13.5px; letter-spacing:-.1px; }
.insp-s { color:var(--muted); font-size:11.5px; margin-bottom:6px; }

.timeline { display:flex; flex-direction:column; padding:6px 0 4px; }
.tl-row { display:grid; grid-template-columns:22px 14px 1fr; gap:8px; padding:7px 0; position:relative; }
.tl-row::before { content:''; position:absolute; left:29px; top:22px; bottom:-8px; width:1px; background:var(--border); }
.tl-row:last-child::before { display:none; }
.tl-n { font:500 10px var(--mono); color:#94A3B8; padding-top:3px; }
.tl-dot { width:9px; height:9px; border-radius:50%; background:#fff; border:1.5px solid #CBD5E1; margin-top:4px; z-index:1; }
.tl-row.done .tl-dot { background:var(--ok); border-color:var(--ok); }
.tl-row.running .tl-dot { background:var(--accent); border-color:var(--accent); animation:pulse 1.1s ease-in-out infinite; }
.tl-row.error .tl-dot { background:var(--err); border-color:var(--err); }
.tl-top { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.tl-name { font-size:12.5px; font-weight:550; }
.tl-row.idle .tl-name { color:#94A3B8; font-weight:500; }
.tl-ms { font:500 11.5px var(--mono); color:var(--muted); }
.tl-row.running .tl-ms { color:var(--accent); }
.tl-row.error .tl-ms { color:var(--err); }
.tl-bar { height:3px; background:#EEF1F6; border-radius:2px; margin-top:5px; overflow:hidden; }
.tl-bar i { display:block; height:100%; background:var(--accent); border-radius:2px; }
.tl-det { font:400 11px var(--mono); color:var(--muted); margin-top:3px; min-height:0; word-break:break-word; }
.tl-det:empty { display:none; }

.tmap { display:flex; gap:1px; height:26px; align-items:stretch; margin-top:20px; }
.tmap i { flex:1; background:#E7ECF5; border-radius:1px; position:relative; min-width:1px; }
.tmap i.hit { background:var(--accent); }
.tmap i u { text-decoration:none; position:absolute; top:-15px; left:50%; transform:translateX(-50%); font:600 9.5px var(--mono); color:var(--accent); }
.tmap-axis { display:flex; justify-content:space-between; font:500 10px var(--mono); color:#94A3B8; margin-top:5px; }

.chunk { border:1px solid var(--border); border-radius:var(--r); padding:10px 12px; margin:10px 0; background:var(--surface); }
.chunk-h { display:flex; align-items:center; gap:8px; }
.rank { width:20px; height:20px; border-radius:6px; background:var(--accent); color:#fff; font:600 11px var(--mono); display:inline-flex; align-items:center; justify-content:center; }
.chunk-id { font:600 12px var(--mono); }
.chunk-ord { margin-left:auto; font:500 10.5px var(--mono); color:var(--muted); }
.score { display:flex; justify-content:space-between; font:500 11px var(--mono); color:var(--muted); margin-top:9px; }
.score b { color:var(--text); }
.score-bar { height:4px; background:#EEF1F6; border-radius:2px; margin-top:4px; overflow:hidden; }
.score-bar i { display:block; height:100%; background:var(--accent2); }
.chunk-t { margin-top:9px; font-size:12.5px; line-height:1.6; color:#2B3650; max-height:170px; overflow:auto; white-space:pre-wrap; padding:8px 10px; background:var(--soft); border-radius:var(--r-sm); border:1px solid #EDF1F7; }

/* ---------- live status / reasoning ---------- */
.live { display:inline-flex; align-items:center; gap:8px; font:500 12px var(--mono); color:var(--accent); padding:4px 10px; border-radius:999px; background:var(--tint); border:1px solid #D7E0FF; margin-bottom:6px; }
.live .pdot { width:7px; height:7px; border-radius:50%; background:var(--accent); animation:pulse 1s ease-in-out infinite; }
.reason { font-size:12.5px; line-height:1.6; color:var(--muted); border-left:2px solid var(--border); padding:2px 0 2px 12px; margin:4px 0 8px; white-space:pre-wrap; max-height:180px; overflow:auto; }
.errbox { border:1px solid #FAD4D0; background:#FEF3F2; color:#7A271A; border-radius:var(--r); padding:10px 14px; font-size:13.5px; }
.errbox b { color:var(--err); }
.warnbox { border:1px solid #F2E2BC; background:#FFF8EA; color:#7A5A12; border-radius:var(--r); padding:8px 12px; font-size:12.5px; margin-top:6px; }

/* ---------- copy button ---------- */
.copybtn { font:500 12.5px var(--ui); padding:5px 11px; border-radius:var(--r-sm); border:1px solid var(--border); background:var(--surface);
  color:var(--text); cursor:pointer; box-shadow:var(--sh); }
.copybtn:hover { background:var(--soft); border-color:#C9D3E6; }
.copybtn.ok { color:var(--ok); border-color:#C5EBDA; background:#ECFBF4; }

/* ---------- empty state / schematic ---------- */
.hero { padding:34px 0 8px; max-width:860px; margin:0 auto; }
.hero-k { font:600 11px var(--mono); letter-spacing:1.4px; color:var(--accent); text-transform:uppercase; }
.hero-t { font-weight:680; font-size:36px; letter-spacing:-1.2px; line-height:1.12; margin:10px 0 10px; }
.hero-t span { color:var(--muted); font-weight:560; }
.hero-p { color:var(--muted); font-size:15px; max-width:560px; line-height:1.6; }
.schematic { max-width:860px; margin:26px auto 8px; border:1px solid var(--border); border-radius:12px; background:var(--surface); padding:14px 16px 16px; box-shadow:var(--sh); }
.sch-cap { font:600 10.5px var(--mono); letter-spacing:1.1px; color:var(--muted); margin-bottom:12px; }
.sch-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; }
.sch-node { border:1px solid var(--border); border-radius:var(--r-sm); padding:8px 10px; background:var(--soft); display:flex; flex-direction:column; gap:1px; }
.sch-node .sch-n { font:500 10px var(--mono); color:var(--accent); }
.sch-node b { font-size:13px; font-weight:600; }
.sch-node em { font:400 10.5px var(--mono); color:var(--muted); font-style:normal; }
.sug-cap { max-width:860px; margin:22px auto 8px; font:600 10.5px var(--mono); letter-spacing:1.1px; color:var(--muted); text-transform:uppercase; }
.st-key-sugs { max-width:860px; margin:0 auto; }

/* ---------- source cards ---------- */
.src { display:grid; grid-template-columns:132px 1fr; gap:14px; align-items:start; }
.src img, .src .thumb { width:132px; height:74px; border-radius:8px; object-fit:cover; background:var(--tint); border:1px solid var(--border); }
.src-t { font-weight:600; font-size:14.5px; letter-spacing:-.15px; line-height:1.3; }
.src-id { font:500 11.5px var(--mono); color:var(--muted); margin:2px 0 8px; }
.src-stats { display:flex; flex-wrap:wrap; gap:6px; }

.hist { width:100%; border-collapse:collapse; font-size:12.5px; }
.hist th { text-align:left; font:600 10.5px var(--mono); letter-spacing:.8px; text-transform:uppercase; color:var(--muted); padding:6px 8px; border-bottom:1px solid var(--border); }
.hist td { padding:8px; border-bottom:1px solid #EEF1F6; font-family:var(--mono); font-size:12px; }
.hist .bar { display:inline-block; height:6px; border-radius:3px; background:var(--accent2); vertical-align:middle; margin-right:6px; }

@media (max-width: 900px) {
  .block-container { padding:.8rem .9rem 6rem !important; }
  .hero-t { font-size:28px; }
  .st-key-inspector { position:static; max-height:none; }
  .src { grid-template-columns:1fr; }
}
</style>
"""

COPY_JS = """
<script>
(function () {
  if (window.__mlxCopyInstalled) return;
  window.__mlxCopyInstalled = true;
  function toTop() {
    try {
      ['[data-testid="stAppScrollToBottomContainer"]', '[data-testid="stMain"]', '[data-testid="stAppViewContainer"]'].forEach(function (q) {
        var el = document.querySelector(q); if (el) el.scrollTo(0, 0);
      });
      window.scrollTo(0, 0);
    } catch (e) {}
  }
  // Streamlit sticks to the bottom when a chat input exists. For the home screen and the
  // Sources/Insights pages we want the top, so scroll up whenever such a page (re)appears.
  var lastMarker = null;
  setInterval(function () {
    var marker = document.querySelector('.hero2, .view-title');
    if (marker && marker !== lastMarker) {
      lastMarker = marker;
      toTop(); setTimeout(toTop, 120); setTimeout(toTop, 450);
    } else if (!marker) { lastMarker = null; }
  }, 250);
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-copy]');
    if (!b) return;
    var text = b.getAttribute('data-copy');
    var label = b.getAttribute('data-label') || 'Copy';
    function ok() {
      b.textContent = 'Copied'; b.classList.add('ok');
      setTimeout(function () { b.textContent = label; b.classList.remove('ok'); }, 1400);
    }
    function fallback() {
      var t = document.createElement('textarea');
      t.value = text; t.style.position = 'fixed'; t.style.opacity = '0';
      document.body.appendChild(t); t.select();
      try { document.execCommand('copy'); ok(); } catch (err) { b.textContent = 'Copy failed'; }
      document.body.removeChild(t);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(ok, fallback);
    } else { fallback(); }
  });
})();
</script>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.html(COPY_JS, unsafe_allow_javascript=True)


# ============================================================
# MOTION + SIMPLIFIED LAYER (v2) — layered over the base design system
# ============================================================

CSS2 = """
<style>
@property --a { syntax:'<angle>'; inherits:false; initial-value:0deg; }
@keyframes spin { to { --a:360deg; } }
@keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:none; } }
@keyframes popIn { from { opacity:0; transform:scale(.94) translateY(8px); } to { opacity:1; transform:none; } }
@keyframes drift { 0% { transform:translate3d(-4%,-3%,0) scale(1); } 100% { transform:translate3d(5%,4%,0) scale(1.15); } }
@keyframes sheen { to { background-position:300% 0; } }
@keyframes travel { 0% { left:-22%; } 100% { left:104%; } }
@keyframes ring { 0% { box-shadow:0 0 0 0 rgba(49,92,235,.45); } 100% { box-shadow:0 0 0 11px rgba(49,92,235,0); } }
@keyframes bounce { 0%,80%,100% { transform:translateY(0); opacity:.45; } 40% { transform:translateY(-6px); opacity:1; } }
@keyframes rot { 0% { opacity:0; transform:translateY(55%); } 5%,22% { opacity:1; transform:none; } 27%,100% { opacity:0; transform:translateY(-55%); } }
@keyframes grow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@keyframes blink { 50% { opacity:0; } }
@keyframes float { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-6px); } }

/* ---------- living background ---------- */
html, body, .stApp { background:#F4F6FA !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background:transparent !important; position:relative; z-index:1; }
[data-testid="stHeader"] { z-index:5; }
[data-testid="stSidebar"] { z-index:3; }
.stApp::before { content:''; position:fixed; inset:-25%; z-index:0; pointer-events:none;
  background:
    radial-gradient(34% 34% at 18% 22%, rgba(49,92,235,.17), transparent 70%),
    radial-gradient(30% 30% at 82% 12%, rgba(8,126,139,.15), transparent 70%),
    radial-gradient(34% 34% at 72% 88%, rgba(139,92,246,.12), transparent 70%),
    radial-gradient(26% 26% at 8% 85%, rgba(49,92,235,.10), transparent 70%);
  animation:drift 24s ease-in-out infinite alternate; }
.block-container { max-width:1040px !important; padding-top:.8rem !important; }

/* ---------- top bar ---------- */
.tb-brand { display:flex; align-items:center; gap:10px; }
.tb-brand .brand-name { font-size:15px; }
.tb-status { display:flex; justify-content:flex-end; }
[data-testid="stButtonGroup"] > div { background:rgba(255,255,255,.88) !important; border:1px solid var(--border) !important; border-radius:999px !important; padding:3px !important; box-shadow:var(--sh); gap:2px; width:fit-content; margin:0 auto; }
button[data-variant="segmented_control"] { border-radius:999px !important; border:0 !important; background:transparent !important; box-shadow:none !important; padding:.3rem 1.15rem !important; transition:all .2s; }
button[data-variant="segmented_control"], button[data-variant="segmented_control"] * { color:var(--muted) !important; font-weight:550 !important; }
button[data-variant="segmented_control"]:hover { background:var(--tint) !important; }
button[data-variant="segmented_control"][aria-checked="true"] { background:var(--text) !important; box-shadow:0 4px 12px rgba(23,32,51,.25) !important; }
button[data-variant="segmented_control"][aria-checked="true"], button[data-variant="segmented_control"][aria-checked="true"] * { color:#fff !important; }

/* ---------- hero ---------- */
.hero2 { text-align:center; padding:6px 0 0; }
.h-pill { display:inline-flex; align-items:center; gap:8px; padding:5px 13px; border-radius:999px; background:rgba(255,255,255,.9); border:1px solid var(--border);
  font:500 11.5px var(--mono); color:var(--muted); box-shadow:var(--sh); animation:fadeUp .6s both; }
.h-pill b { width:7px; height:7px; border-radius:50%; background:var(--ok); animation:ring 1.6s ease-out infinite; box-shadow:0 0 0 0 rgba(22,132,93,.4); }
.h-title { font-weight:800; font-size:clamp(30px,4.4vw,52px); line-height:1.05; letter-spacing:-2px; margin:14px 0 8px; animation:fadeUp .7s .06s both; }
.rot { display:inline-grid; vertical-align:bottom; }
.rot span { grid-area:1/1; opacity:0; animation:rot 12s infinite; padding:0 .08em;
  background:linear-gradient(90deg,#315CEB,#087E8B,#8B5CF6,#315CEB); background-size:300% 100%; -webkit-background-clip:text; background-clip:text;
  color:transparent !important; -webkit-text-fill-color:transparent; animation-name:rot, sheen; animation-duration:12s, 6s; animation-iteration-count:infinite, infinite; animation-timing-function:ease, linear; }
.rot span:nth-child(2) { animation-delay:3s, 0s; } .rot span:nth-child(3) { animation-delay:6s, 0s; } .rot span:nth-child(4) { animation-delay:9s, 0s; }
.h-sub { color:var(--muted); font-size:16px; line-height:1.6; max-width:520px; margin:0 auto; animation:fadeUp .8s .12s both; }

/* ---------- pipeline ribbon ---------- */
.rib { position:relative; margin:14px auto 6px; max-width:940px; animation:fadeUp .9s .2s both; }
.rib.still { animation:none; margin:10px 0 6px; }
.rib-track { position:absolute; left:5.55%; right:5.55%; top:14px; height:3px; background:#E1E7F1; border-radius:2px; overflow:hidden; }
.rib-fill { position:absolute; inset:0 auto 0 0; background:linear-gradient(90deg,#315CEB,#087E8B); border-radius:2px; transition:width .4s; }
.rib.demo .rib-track::after { content:''; position:absolute; top:0; bottom:0; width:20%; background:linear-gradient(90deg,transparent,#315CEB,transparent); animation:travel 3.4s ease-in-out infinite; }
.rib-nodes { position:relative; display:grid; grid-template-columns:repeat(9,1fr); }
.rn { display:flex; flex-direction:column; align-items:center; gap:5px; text-align:center; }
.rd { width:30px; height:30px; border-radius:50%; background:#fff; border:2px solid #D5DCEA; display:grid; place-items:center; font:600 10.5px var(--mono); color:#94A3B8; position:relative; z-index:1; transition:all .25s; }
.rib.demo .rn:hover .rd { transform:translateY(-4px) scale(1.1); border-color:var(--accent); color:var(--accent); }
.rn.done .rd { background:var(--ok); border-color:var(--ok); color:#fff; }
.rn.running .rd { border-color:var(--accent); color:var(--accent); animation:ring 1.2s ease-out infinite; }
.rn.error .rd { background:var(--err); border-color:var(--err); color:#fff; }
.rl { font:600 11.5px var(--ui); color:var(--text); }
.rn.idle .rl { color:#7C8AA5; }
.rt { font:500 10px var(--mono); color:var(--muted); min-height:13px; }
.rn.running .rt { color:var(--accent); }
.rn.error .rt { color:var(--err); }
.rib-note { text-align:center; font:500 11.5px var(--mono); color:var(--muted); margin-top:6px; }

/* ---------- onboarding card (spinning border) ---------- */
.st-key-onboard { max-width:640px; margin:22px auto 0; padding:18px 18px 14px; border:2px solid transparent; border-radius:18px; box-shadow:var(--sh-lg);
  background:linear-gradient(#fff,#fff) padding-box, conic-gradient(from var(--a), #315CEB, #087E8B, #8B5CF6, #315CEB) border-box; animation:spin 7s linear infinite, fadeUp .8s .3s both; }
.ob-t { font-weight:650; font-size:15px; margin-bottom:10px; }
.ob-s { font:500 11.5px var(--mono); color:var(--muted); margin-top:8px; }

/* ---------- suggestion tiles ---------- */
.sug-cap { text-align:center; margin:26px 0 10px; }
.st-key-sugs { max-width:720px; margin:0 auto; }
.st-key-sugs button { min-height:70px !important; justify-content:flex-start !important; text-align:left !important; padding:.8rem 1rem !important; border-radius:14px !important;
  border:1.5px solid var(--border) !important; background:rgba(255,255,255,.92) !important; animation:popIn .5s both; transition:transform .2s, box-shadow .2s !important; }
.st-key-sugs [data-testid="stColumn"]:nth-child(2) button { animation-delay:.08s; }
.st-key-sugs button:hover { transform:translateY(-4px) rotate(-.4deg); box-shadow:0 12px 28px rgba(49,92,235,.16) !important;
  border:1.5px solid transparent !important; background:linear-gradient(#fff,#fff) padding-box, conic-gradient(from var(--a), #315CEB, #087E8B, #8B5CF6, #315CEB) border-box !important; animation:spin 3s linear infinite !important; }
.st-key-sugs button p { font-size:14.5px !important; font-weight:550 !important; }

/* ---------- conversation ---------- */
.conv-bar { display:flex; align-items:center; gap:10px; min-height:40px; animation:fadeUp .4s both; }
.conv-bar .conv-title { font-size:15px; }
[data-testid="stChatMessage"] { animation:fadeUp .45s ease both; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:transparent !important; border:0 !important; padding:.35rem 0 !important; flex-direction:row-reverse; margin:.3rem auto !important; }
[data-testid="stChatMessageAvatarUser"] { display:none !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { flex:0 1 auto; width:fit-content; max-width:82%;
  background:linear-gradient(135deg,#315CEB,#3F6DF5); border-radius:20px 20px 5px 20px; padding:.6rem 1.05rem; box-shadow:0 6px 18px rgba(49,92,235,.25); animation:popIn .35s both; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * { color:#fff !important; }
[data-testid="stChatMessageAvatarAssistant"] { background:#fff !important; border:1px solid var(--border); box-shadow:var(--sh); animation:float 4s ease-in-out infinite; }

.think { display:inline-flex; align-items:center; gap:5px; padding:7px 13px; border-radius:999px; background:rgba(255,255,255,.95); border:1px solid #D7E0FF; box-shadow:var(--sh); }
.think i { width:7px; height:7px; border-radius:50%; background:var(--accent); animation:bounce 1.1s infinite; }
.think i:nth-child(2) { animation-delay:.15s; } .think i:nth-child(3) { animation-delay:.3s; }
.think span { font:500 12px var(--mono); color:var(--accent); margin-left:6px; }
.cursor { display:inline-block; width:8px; height:1em; background:var(--accent); vertical-align:-2px; margin-left:2px; border-radius:2px; animation:blink 1s steps(2) infinite; }

.minimeta { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 4px; }
.how-hint { font:500 11.5px var(--mono); color:var(--muted); }
details.pp { margin-top:12px; border:1px solid var(--border); border-radius:var(--r); background:var(--soft); }
details.pp summary { cursor:pointer; padding:8px 12px; font:600 11.5px var(--mono); color:var(--muted); }
details.pp pre { margin:0; padding:10px 12px; white-space:pre-wrap; font:400 11.5px/1.6 var(--mono); color:#2B3650; max-height:260px; overflow:auto; border-top:1px solid var(--border); background:transparent; }

/* ---------- chat input ---------- */
[data-testid="stChatInput"] { max-width:760px; border-radius:18px !important; transition:box-shadow .25s; }
[data-testid="stChatInput"]:focus-within { border:1.5px solid transparent !important; animation:spin 4s linear infinite;
  background:linear-gradient(#fff,#fff) padding-box, conic-gradient(from var(--a), #315CEB, #087E8B, #8B5CF6, #315CEB) border-box !important; box-shadow:0 10px 32px rgba(49,92,235,.18) !important; }

/* ---------- buttons / tiles motion ---------- */
.stButton > button { transition:transform .15s, box-shadow .2s, background .15s !important; }
.stButton > button:hover { transform:translateY(-1px); }
.stButton > button:active { transform:scale(.97); }
.tiles .tile { animation:fadeUp .5s both; transition:transform .2s, box-shadow .2s; }
.tiles .tile:hover { transform:translateY(-3px); box-shadow:var(--sh-lg); }
.tiles .tile:nth-child(2) { animation-delay:.05s; } .tiles .tile:nth-child(3) { animation-delay:.1s; }
.tiles .tile:nth-child(4) { animation-delay:.15s; } .tiles .tile:nth-child(5) { animation-delay:.2s; } .tiles .tile:nth-child(6) { animation-delay:.25s; }
.stack i { transform-origin:left; animation:grow .9s cubic-bezier(.2,.8,.2,1) both; }
.panel { animation:fadeUp .5s both; background:rgba(255,255,255,.92); }
.chunk { animation:fadeUp .5s both; transition:transform .2s, box-shadow .2s; }
.chunk:hover { transform:translateY(-2px); box-shadow:var(--sh-lg); }
.score-bar i { animation:grow .8s both; transform-origin:left; }
.tmap i.hit { animation:popIn .5s both; }
.view-title { animation:fadeUp .4s both; }

/* sidebar */
[data-testid="stSidebar"] { background:rgba(255,255,255,.95) !important; }


@keyframes marq { to { transform:translateX(-50%); } }
@keyframes hop { 0% { left:0; opacity:0; } 15% { opacity:1; } 85% { opacity:1; } 100% { left:calc(100% - 10px); opacity:0; } }
@keyframes glow { 0%,100% { box-shadow:0 6px 20px rgba(49,92,235,.10); } 50% { box-shadow:0 10px 30px rgba(8,126,139,.22); } }

.ticker { overflow:hidden; max-width:780px; margin:12px auto 0; -webkit-mask-image:linear-gradient(90deg,transparent,#000 14%,#000 86%,transparent); mask-image:linear-gradient(90deg,transparent,#000 14%,#000 86%,transparent); animation:fadeUp .9s .2s both; }
.ticker-track { display:flex; width:max-content; animation:marq 42s linear infinite; }
.ticker-track span { margin-right:36px; font:600 19px var(--ui); color:#95A1BA; white-space:nowrap; }
.ticker-track span:nth-child(4n+1) { color:var(--accent); } .ticker-track span:nth-child(4n+3) { color:var(--accent2); }

.bridge { display:flex; align-items:stretch; justify-content:center; gap:10px; margin:14px 0 10px; }
.bnode { flex:1; max-width:280px; position:relative; overflow:hidden; text-align:left; padding:10px 14px 10px 17px; border:1px solid var(--border); border-radius:16px;
  background:rgba(255,255,255,.96); animation:glow 5s ease-in-out infinite, fadeUp .6s both; }
.bnode::before { content:''; position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(180deg,#315CEB,#8B5CF6); }
.bnode.ans::before { background:linear-gradient(180deg,#087E8B,#16845D); }
.bnode.ans { animation-delay:1.2s, .1s; }
.bk { display:block; font:600 10px var(--mono); letter-spacing:1.1px; text-transform:uppercase; color:var(--muted); }
.bg { display:block; font-weight:750; font-size:26px; letter-spacing:-.6px; line-height:1.3; color:var(--text); }
.bn { display:block; font:500 11.5px var(--mono); color:var(--muted); }
.bridge-arrow { align-self:center; position:relative; width:84px; height:2px; background:repeating-linear-gradient(90deg,#B9C6F5 0 6px,transparent 6px 12px); }
.bridge-arrow i { position:absolute; top:-4px; left:0; width:10px; height:10px; border-radius:50%; background:var(--accent); animation:hop 1.9s ease-in-out infinite; }
.bridge-arrow::after { content:''; position:absolute; right:-3px; top:-5px; border-left:9px solid var(--accent2); border-top:6px solid transparent; border-bottom:6px solid transparent; }
.bridge-note { text-align:center; font:500 11.5px var(--mono); color:var(--muted); margin:2px 0 6px; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { border:1px solid var(--border) !important; }
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] > div { background:var(--surface) !important; background-color:var(--surface) !important; border:1px solid var(--border); border-radius:var(--r); box-shadow:var(--sh-lg); }
[data-testid="stSelectboxVirtualDropdown"] [role="option"], [data-testid="stSelectboxVirtualDropdown"] [role="option"] * { color:var(--text) !important; -webkit-text-fill-color:var(--text) !important; background-color:transparent !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover, [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] { background-color:var(--tint) !important; }
[data-testid="stToast"], [data-testid="stToast"] > div, [data-testid="stToastContainer"] [role="alert"] { background:var(--surface) !important; background-color:var(--surface) !important; border:1px solid var(--border) !important; border-radius:12px !important; box-shadow:var(--sh-lg) !important; }
[data-testid="stToast"], [data-testid="stToast"] *, [data-testid="stToastContainer"] [role="alert"] * { color:var(--text) !important; -webkit-text-fill-color:var(--text) !important; }
.st-key-langbox { max-width:720px; margin:6px auto 0; }
.tracks .srcs { margin-top:0; }
.kv.tight { grid-template-columns:130px 1fr; margin:6px 0 10px; }
@media (max-width:760px) { .bnode .bg { font-size:20px; } .bridge-arrow { width:36px; } }
@media (max-width:760px) { .rl { display:none; } .rt { font-size:9px; } .h-title { letter-spacing:-1.4px; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation:none !important; transition:none !important; } }
</style>
"""
st.markdown(CSS2, unsafe_allow_html=True)


# ============================================================
# SESSION + CONVERSATION STATE
# ============================================================

store = get_store()
ss = st.session_state

for _k, _v in {
    "view": "Chat",
    "conv_id": None,
    "max_tokens": 1024,
    "show_reasoning": False,
    "pending": None,
    "video_input": DEFAULT_VIDEO,
    "flash": None,
    "video_lang": "auto",
    "answer_lang": "en",
    "bridge_boost": True,
    "run_index": False,
    "last_tracks": None,
}.items():
    ss.setdefault(_k, _v)

# keep widget-bound preferences alive on views where the widget isn't rendered
for _k in ("max_tokens", "show_reasoning", "video_input", "video_lang", "answer_lang", "bridge_boost"):
    ss[_k] = ss[_k]

DEFAULT_TITLE = "New conversation"
VIEWS = ["Chat", "Sources", "Insights"]


def index_ids():
    return sorted(store["indexes"].keys(), key=lambda v: store["indexes"][v]["built_at"], reverse=True)


def source_label(key):
    rec = store["indexes"].get(key)
    if not rec:
        return key
    return f'{rec.get("title") or rec["video_id"]} · {rec["lang_name"]}'


def new_conv(video_id=None):
    cid = uuid.uuid4().hex[:10]
    if video_id is None:
        ids = index_ids()
        video_id = ids[0] if ids else None
    store["convs"][cid] = {
        "id": cid, "title": DEFAULT_TITLE, "video_id": video_id,
        "messages": [], "created": time.time(), "updated": time.time(),
    }
    ss.conv_id = cid
    return store["convs"][cid]


def current_conv():
    cid = ss.conv_id
    if cid not in store["convs"]:
        for c in sorted(store["convs"].values(), key=lambda c: c["updated"], reverse=True):
            if not c["messages"]:
                ss.conv_id = c["id"]
                return c
        return new_conv()
    return store["convs"][cid]


def go(view):
    ss.view = view


def cb_new_chat():
    conv = store["convs"].get(ss.conv_id)
    if conv and not conv["messages"]:
        ss.view = "Chat"
        return
    new_conv(conv["video_id"] if conv and conv["video_id"] else None)
    ss.view = "Chat"


def cb_open_conv(cid):
    ss.conv_id = cid
    ss.view = "Chat"


def cb_rename(cid):
    title = (ss.get(f"rn_{cid}") or "").strip()
    if title and cid in store["convs"]:
        store["convs"][cid]["title"] = title[:80]


def cb_clear(cid):
    if cid in store["convs"]:
        store["convs"][cid]["messages"] = []
        store["convs"][cid]["title"] = DEFAULT_TITLE


def cb_delete(cid):
    store["convs"].pop(cid, None)
    ss.conv_id = None


def cb_switch_source():
    vid = ss.get("src_select")
    conv = store["convs"].get(ss.conv_id)
    if not vid or not conv or conv["video_id"] == vid:
        return
    if conv["messages"]:
        new_conv(vid)          # never mix retrieval contexts inside one conversation
    else:
        conv["video_id"] = vid


def bind_source(video_id):
    conv = store["convs"].get(ss.conv_id)
    if conv and conv["messages"] and conv["video_id"] != video_id:
        new_conv(video_id)
    elif conv:
        conv["video_id"] = video_id
    else:
        new_conv(video_id)


def cb_use_source(video_id):
    bind_source(video_id)
    ss.view = "Chat"


def cb_remove_source(video_id):
    store["indexes"].pop(video_id, None)
    for c in store["convs"].values():
        if c["video_id"] == video_id:
            c["video_id"] = None
    gc.collect()


def cb_suggest(text):
    ss.pending = {"q": text, "regen": False}


def cb_regen(cid):
    conv = store["convs"].get(cid)
    if not conv:
        return
    msgs = conv["messages"]
    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    last_user = next((m for m in reversed(msgs) if m["role"] == "user"), None)
    if last_user:
        ss.pending = {"q": last_user["content"], "regen": True}


def cb_wipe_all():
    store["convs"].clear()
    store["perf"].clear()
    ss.conv_id = None


def cb_unload():
    unload_llm()


def cb_pick_lang(code):
    ss.video_lang = code
    ss.run_index = True


conv = current_conv()


# ============================================================
# RIBBON — the live, animated pipeline (real statuses + timings)
# ============================================================

RIB_LABELS = {
    "fetch": "Fetch", "split": "Split", "embed": "Embed", "faiss": "Index", "qembed": "Query",
    "mmr": "MMR", "context": "Context", "infer": "MLX", "answer": "Answer",
}


def merged_stages(rec, qstages):
    out = []
    idle = {"status": "idle", "ms": None, "detail": ""}
    for key, label, kind in STAGES:
        if kind == "idx":
            if rec and key in rec["timings"]:
                s = {"status": "done", "ms": rec["timings"][key] * 1000, "detail": rec["details"].get(key, "")}
            else:
                s = idle
        else:
            s = (qstages or {}).get(key) or idle
        out.append((key, label, s))
    return out


def ribbon_html(stages, demo=False, still=True):
    """stages: list of (key, label, state). Node colours/timings are real states."""
    n = len(stages)
    furthest = -1
    for i, (_, _, s) in enumerate(stages):
        if s["status"] in ("done", "running", "error"):
            furthest = i
    fill = 0 if furthest < 0 else furthest / (n - 1) * 100
    nodes = []
    for i, (key, _, s) in enumerate(stages, start=1):
        status = s["status"]
        glyph = {"done": "✓", "error": "!"}.get(status, str(i))
        if status == "done":
            t = fmt_ms(s["ms"]) if key != "answer" else fmt_ms(s["ms"])
        elif status == "running":
            t = "live"
        else:
            t = ""
        nodes.append(
            f'<div class="rn {status}" title="{esc(s["detail"])}"><span class="rd">{glyph}</span>'
            f'<span class="rl">{RIB_LABELS[key]}</span><span class="rt">{t}</span></div>'
        )
    cls = "rib" + (" demo" if demo else "") + (" still" if still else "")
    return (f'<div class="{cls}"><div class="rib-track"><i class="rib-fill" style="width:{fill:.1f}%"></i></div>'
            f'<div class="rib-nodes">{"".join(nodes)}</div></div>')


def idle_stages():
    return [(k, l, {"status": "idle", "ms": None, "detail": ""}) for k, l, _ in STAGES]


# ============================================================
# TOP BAR + SIDEBAR
# ============================================================

with st.sidebar:
    st.html('<div class="side-label" style="padding-top:4px">Conversations</div>')
    st.button("New conversation", key="newchat", icon=":material/add:", on_click=cb_new_chat, width="stretch")
    convs_sorted = sorted(store["convs"].values(), key=lambda c: c["updated"], reverse=True)
    shown = [c for c in convs_sorted if c["messages"]][:20]
    if not shown:
        st.html('<div class="empty-note" style="padding:8px 6px">Your chats will appear here.</div>')
    for c in shown:
        on = c["id"] == conv["id"] and ss.view == "Chat"
        st.button(c["title"], key=f"{'convon' if on else 'conv'}_{c['id']}", icon=":material/chat_bubble_outline:",
                  on_click=cb_open_conv, args=(c["id"],), width="stretch")
    st.html('<div class="side-label">Runtime</div>')
    llm_state = "on" if "llm_load_s" in store else ""
    emb_state = "on" if "embed_load_s" in store else ""
    st.html(
        '<div class="side-status">'
        f'<div class="row"><span><i class="dot {llm_state}"></i>Qwen3-8B · 4-bit</span><span>{"ready" if llm_state else "idle"}</span></div>'
        f'<div class="row"><span><i class="dot {emb_state}"></i>Qwen3-Embedding</span><span>{"ready" if emb_state else "idle"}</span></div>'
        f'<div class="row"><span><i class="dot on"></i>FAISS · MMR</span><span>{len(store["indexes"])} source(s)</span></div>'
        '</div>'
    )

tb1, tb2, tb3 = st.columns([1.4, 2, 1.4], vertical_alignment="center")
with tb1:
    st.html('<div class="tb-brand"><div class="brand-mark">MX</div>'
            '<div><div class="brand-name">MLX RAG Studio</div><div class="brand-sub">local · apple silicon</div></div></div>')
with tb2:
    st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed")
with tb3:
    ready = "llm_load_s" in store
    st.html(f'<div class="tb-status">{chip("● model ready" if ready else "○ model idle", "ok" if ready else "")}</div>')
VIEW = ss.view if ss.view in VIEWS else "Chat"


# ============================================================
# INDEXING (UI wrapper around build_index)
# ============================================================

def valid_video_id(raw_input):
    video_id = extract_video_id(raw_input or "")
    if not video_id:
        st.error("Paste a YouTube URL or an 11-character video ID.")
        return None
    if not VIDEO_ID_RE.match(video_id):
        st.error("That doesn't look like a valid YouTube URL or video ID. "
                 "IDs are exactly 11 characters (letters, digits, - and _).")
        return None
    return video_id


def tracks_html(tracks, chosen=None):
    n_manual = sum(1 for t in tracks if not t["generated"])
    n_auto = len(tracks) - n_manual
    chips = "".join(
        chip(f'{esc(t["name"])} · {esc(t["code"])}' + (" · auto" if t["generated"] else ""),
             "acc" if chosen and t["code"] == chosen else "")
        for t in tracks
    )
    return (f'<div class="tracks"><div class="sec" style="margin:12px 0 6px">Caption languages on this video · '
            f'{n_manual} manual · {n_auto} auto-generated</div><div class="srcs">{chips}</div></div>')


def check_captions(raw_input):
    video_id = valid_video_id(raw_input)
    if not video_id:
        return
    try:
        with st.spinner("Checking which caption languages exist…"):
            tracks = list_transcript_tracks(video_id)
    except Exception as exc:
        log.warning("Caption listing failed for %s: %s", video_id, type(exc).__name__)
        st.error(friendly_error(exc))
        return
    ss.last_tracks = {"video_id": video_id, "tracks": tracks}


def index_video_ui(raw_input):
    video_id = valid_video_id(raw_input)
    if not video_id:
        return
    requested = ss.video_lang
    try:
        tracks = list_transcript_tracks(video_id)
    except Exception as exc:
        log.warning("Caption listing failed for %s: %s", video_id, type(exc).__name__)
        st.error(friendly_error(exc))
        return
    ss.last_tracks = {"video_id": video_id, "tracks": tracks}
    try:
        track, reason = pick_track(tracks, requested)
    except LanguageUnavailable as exc:
        st.error(str(exc))
        st.html('<div class="ob-s">YouTube can machine-translate captions for some videos, but this app only indexes '
                'caption tracks that really exist. Pick one below, or switch the video language to Auto-detect:</div>')
        seen, opts = set(), []
        for t in sorted(tracks, key=lambda t: t["generated"]):
            if t["code"] not in seen:
                seen.add(t["code"])
                opts.append(t)
        cols = st.columns(5)
        for i, t in enumerate(opts[:35]):
            with cols[i % 5]:
                st.button(t["name"], key=f"alt_{t['code']}", on_click=cb_pick_lang, args=(t["code"],), width="stretch")
        return

    key = index_key(video_id, track["code"])
    if key in store["indexes"]:
        st.toast(f"{track['name']} index for this video already exists — reusing it.", icon=":material/check_circle:")
        bind_source(key)
        st.rerun()
        return
    with store["lock"]:
        if key in store["building"]:
            st.warning("This video is already being indexed in another session.")
            return
        store["building"].add(key)

    state = {k: {"status": "idle", "ms": None, "detail": ""} for k, _, _ in STAGES}
    ph = st.empty()
    note = st.empty()

    def draw():
        ph.html(ribbon_html([(k, l, state[k]) for k, l, _ in STAGES], still=True))

    try:
        draw()
        kind = "auto-generated" if track["generated"] else "manual"
        note.html(f'<div class="rib-note">Indexing {esc(track["name"])} ({kind} captions) · {esc(reason)}. '
                  'The first run also loads the embedding model.</div>')
        rec = None
        for ev in build_index(video_id, track, reason, tracks):
            if ev[0] == "stage":
                _, skey, st_, ms, detail = ev
                state[skey] = {"status": st_, "ms": ms, "detail": detail}
                draw()
            elif ev[0] == "done":
                rec = ev[1]
        store["indexes"][key] = rec
        log.info("Indexed %s (%s): %d chunks, %.2fs", video_id, track["code"], rec["n_chunks"], rec["total_s"])
        bind_source(key)
        ss.flash = f"Ready — {track['name']} index with {rec['n_chunks']:,} chunks in {fmt_s(rec['total_s'])}."
    except Exception as exc:
        log.exception("Indexing failed for %s", video_id)
        note.empty()
        st.error(friendly_error(exc))
        return
    finally:
        with store["lock"]:
            store["building"].discard(key)
    st.rerun()


# ============================================================
# CHAT VIEW
# ============================================================

def meta_line(trace):
    bits = []
    vl = trace.get("video_lang")
    al = LANG_BY_CODE.get(trace.get("answer_lang", "en"), LANG_BY_CODE["en"])["name"]
    if vl:
        bits.append(chip(f'{esc(vl)} → {esc(al)}', "ok"))
    if trace.get("q_script") and trace.get("v_script") and trace["q_script"] != trace["v_script"]:
        bits.append(chip(f'{esc(trace["q_script"])} question · cross-language', "acc"))
    if trace.get("bridge_query"):
        bits.append(chip(f'retrieval boost +{fmt_s(trace["bridge_s"])}', "warn"))
    if trace.get("total_s") is not None:
        bits.append(chip(fmt_s(trace["total_s"]), "acc"))
    if trace.get("tps"):
        bits.append(chip(f'{trace["tps"]:.1f} tok/s'))
    if trace.get("tokens") is not None:
        bits.append(chip(f'{trace["tokens"]} tokens'))
    if trace.get("ttft_s") is not None:
        bits.append(chip(f'first token {fmt_s(trace["ttft_s"])}'))
    return '<div class="minimeta">' + "".join(bits) + "</div>"


def how_html(rec, trace):
    parts = [ribbon_html(merged_stages(rec, trace["stages"]), still=True)]
    al = LANG_BY_CODE.get(trace.get("answer_lang", "en"), LANG_BY_CODE["en"])
    rows = [("Video language", esc(f'{trace.get("video_lang", "—")}' + (f' · {rec["lang_kind"]} captions · {rec["lang_reason"]}' if rec else ""))),
            ("Answer language", esc(f'{al["name"]} ({al["native"]})')),
            ("Question script", esc(f'{trace.get("q_script", "—")} (transcript script: {trace.get("v_script", "—")})')),
            ("Retrieval query", esc(trace["bridge_query"]) + " <em>(translated by Qwen3-8B)</em>" if trace.get("bridge_query") else "the original question")]
    kv = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    parts.append(f'<div class="sec" style="margin-bottom:0">Languages</div><dl class="kv tight">{kv}</dl>')
    if rec:
        hits = [(s["rank"], s["chunk"]) for s in trace["sources"]]
        parts.append('<div class="sec" style="margin-bottom:0">Where the evidence sits</div>'
                     + transcript_map_html(rec["n_chunks"], hits))
    parts.append('<div class="sec" style="margin-bottom:0">Retrieved chunks (original transcript language)</div>')
    parts.append(chunk_cards_html(trace["sources"], trace["scores_available"]))
    parts.append(f'<details class="pp"><summary>Prompt sent to the model</summary><pre>{esc(trace["prompt"])}</pre></details>')
    return "".join(parts)


def render_assistant(msg, is_last, conv):
    if msg.get("error"):
        st.html(f'<div class="errbox"><b>Could not generate an answer.</b><br>{esc(msg["error"])}</div>')
        if is_last:
            st.button("Try again", key=f"retry_{msg['id']}", icon=":material/refresh:",
                      on_click=cb_regen, args=(conv["id"],))
        return
    if msg.get("reasoning") and ss.show_reasoning:
        with st.expander("Reasoning", expanded=False):
            st.html(f'<div class="reason">{esc(msg["reasoning"])}</div>')
    if msg["content"]:
        st.markdown(msg["content"])
    trace = msg.get("trace") or {}
    if trace.get("finish_reason") == "length":
        st.html('<div class="warnbox">The response hit the maximum length. '
                'Raise it in Insights → Models to allow longer answers.</div>')
    if trace:
        st.html(meta_line(trace))
    c1, c2, _sp = st.columns([0.9, 1.3, 6], gap="small", vertical_alignment="center")
    with c1:
        st.html(f'<button class="copybtn" data-label="Copy" data-copy="{esc(msg["content"], quote=True)}">Copy</button>')
    with c2:
        if is_last:
            st.button("Retry", key=f"regen_{msg['id']}", icon=":material/refresh:",
                      on_click=cb_regen, args=(conv["id"],), help="Regenerate this answer")
    if trace:
        rec = store["indexes"].get(trace.get("index_key"))
        n_src = len(trace.get("sources", []))
        with st.expander(f"How this was answered · {n_src} sources · {fmt_s(trace.get('total_s'))}"):
            st.html(how_html(rec, trace))


def render_history(conv):
    msgs = conv["messages"]
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])
        else:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                render_assistant(m, i == len(msgs) - 1, conv)


def stream_answer(conv, rec, question, replace_last):
    """Run one RAG turn with live streaming. Persists the result in the store."""
    qstages = empty_stage_state()
    raw = ""
    trace = None
    t_start = time.perf_counter()
    status_text = "Embedding your question"

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        rib_ph = st.empty()
        live_ph = st.empty()
        reason_ph = st.empty()
        text_ph = st.empty()

        def draw_ribbon():
            rib_ph.html(ribbon_html(merged_stages(rec, qstages), still=True))

        def draw_status():
            el = time.perf_counter() - t_start
            live_ph.html(f'<div class="think"><i></i><i></i><i></i><span>{esc(status_text)} · {el:.1f}s</span></div>')

        draw_ribbon()
        draw_status()
        last_draw = 0.0
        try:
            for ev in run_query(rec, question, ss.max_tokens, ss.answer_lang, ss.bridge_boost):
                kind = ev[0]
                if kind == "stage":
                    _, key, st_, ms, detail = ev
                    qstages[key] = {"status": st_, "ms": ms, "detail": detail}
                    draw_ribbon()
                    names = {"qembed": "Embedding your question", "mmr": "Finding the best passages",
                             "context": "Building context", "infer": "Thinking"}
                    if st_ == "running" and key in names:
                        status_text = names[key] if not detail or detail == "streaming" else f"{names[key]} · {detail}"
                    draw_status()
                elif kind == "token":
                    raw += ev[1]
                    now = time.perf_counter()
                    if now - last_draw > 0.08:
                        last_draw = now
                        reasoning, answer, thinking = split_reasoning(raw)
                        if thinking:
                            status_text = "Reasoning"
                            if ss.show_reasoning:
                                reason_ph.html(f'<div class="reason">{esc(reasoning[-700:])}</div>')
                        else:
                            status_text = "Writing"
                            reason_ph.empty()
                        draw_status()
                        text_ph.markdown(answer + ' <span class="cursor"></span>' if answer else "", unsafe_allow_html=True)
                elif kind == "done":
                    trace = ev[1]
                    trace["stages"] = {k: dict(v) for k, v in qstages.items()}
        except Exception as exc:
            log.exception("Query failed")
            msg = friendly_error(exc)
            conv["messages"].append({"id": uuid.uuid4().hex[:10], "role": "assistant",
                                     "content": "", "error": msg})
            conv["updated"] = time.time()
            st.rerun()
            return

        reasoning, answer, thinking = split_reasoning(raw)
        note = ""
        if thinking and not answer:
            note = ("The model used its entire token budget on reasoning before answering. "
                    "Raise the maximum response length in Insights → Models and retry.")
        elif not answer:
            note = "The model returned an empty response."
        if note:
            trace["finish_reason"] = "length"
        live_ph.empty()
        reason_ph.empty()
        text_ph.markdown(answer or note)

    conv["messages"].append({
        "id": uuid.uuid4().hex[:10], "role": "assistant",
        "content": answer or note, "reasoning": reasoning, "trace": trace,
    })
    conv["updated"] = time.time()
    st.rerun()


SUGGESTIONS = [
    "Summarize the main ideas",
    "Explain the key concepts",
    "What examples are discussed?",
    "What does the speaker conclude?",
]


def fmt_video_lang(code):
    if code == "auto":
        return "Auto-detect (recommended)"
    if code == "other":
        return "Other / any available language"
    i = LANG_BY_CODE[code]
    return i["name"] if i["native"] == i["name"] else f'{i["native"]} · {i["name"]}'


def fmt_answer_lang(code):
    i = LANG_BY_CODE[code]
    return i["name"] if i["native"] == i["name"] else f'{i["native"]} · {i["name"]}'


def fmt_answer_compact(code):
    return f'Answer in {LANG_BY_CODE[code]["name"]}'


def bridge_preview(rec=None):
    a = LANG_BY_CODE.get(ss.answer_lang, LANG_BY_CODE["en"])
    if rec:
        vi = lang_info(rec["lang_code"], rec["lang_name"])
        vbig, vsub = vi["native"], f'{rec["lang_name"]} · {rec["lang_kind"]} captions'
    else:
        v = ss.video_lang
        if v == "auto":
            vbig, vsub = "Auto", "detects the spoken language"
        elif v == "other":
            vbig, vsub = "Any", "uses the best available captions"
        else:
            vi = LANG_BY_CODE[v]
            vbig, vsub = vi["native"], f'{vi["name"]} captions'
    return (f'<div class="bridge"><div class="bnode"><span class="bk">Video language</span>'
            f'<span class="bg">{esc(vbig)}</span><span class="bn">{esc(vsub)}</span></div>'
            f'<div class="bridge-arrow"><i></i></div>'
            f'<div class="bnode ans"><span class="bk">Answer language</span><span class="bg">{esc(a["native"])}</span>'
            f'<span class="bn">{esc(a["name"])} · chosen independently</span></div></div>')


def lang_bridge(rec=None):
    """Two independent language controls: what the video is in vs. what you want to read."""
    st.html(bridge_preview(rec))
    if rec:
        st.selectbox("Answer language", ANSWER_LANG_OPTIONS, key="answer_lang", format_func=fmt_answer_lang)
        st.html(f'<div class="bridge-note">Captions: {esc(rec["lang_reason"])}. '
                'Changing the answer language never re-indexes the video.</div>')
    else:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.selectbox("Video language", VIDEO_LANG_OPTIONS, key="video_lang", format_func=fmt_video_lang)
        with c2:
            st.selectbox("Answer language", ANSWER_LANG_OPTIONS, key="answer_lang", format_func=fmt_answer_lang)


def onboard_card(title):
    with st.container(key="onboard"):
        st.html(f'<div class="ob-t">{esc(title)}</div>')
        a, b, c = st.columns([4, 1.6, 1.3], vertical_alignment="center")
        with a:
            raw = st.text_input("YouTube link", key="video_input", label_visibility="collapsed",
                                placeholder="https://youtu.be/…  or an 11-character video ID")
        with b:
            check = st.button("Captions", key="ob_check", icon=":material/subtitles:", width="stretch",
                              help="List the caption languages YouTube really has for this video")
        with c:
            go_btn = st.button("Index", key="ob_index", type="primary", icon=":material/bolt:", width="stretch")
        lang_bridge()
        st.html('<div class="ob-s">Any caption language YouTube offers · chunks 1000/200 · one index per video + language, reused next time</div>')
        run = go_btn or ss.pop("run_index", False)
        if check:
            check_captions(raw)
        if run:
            index_video_ui(raw)
        lt = ss.last_tracks
        if lt and lt["video_id"] == extract_video_id(raw or ""):
            st.html(tracks_html(lt["tracks"]))


TICKER = ["Hello", "నమస్కారం", "नमस्ते", "வணக்கம்", "こんにちは", "Hola", "مرحبا", "你好", "Bonjour", "Guten Tag", "안녕하세요", "Olá", "Привет", "ನಮಸ್ಕಾರ", "നമസ്കാരം", "नमस्कार", "নমস্কার", "નમસ્તે", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "سلام"]


def ticker_html():
    items = "".join(f"<span>{esc(w)}</span>" for w in TICKER)
    return f'<div class="ticker"><div class="ticker-track">{items}{items}</div></div>'


def render_chat_bar(conv, rec):
    c1, c2, c3 = st.columns([5, 2.7, 1], vertical_alignment="center", gap="small")
    with c1:
        pill = ""
        if rec:
            pill = chip(esc(rec.get("title") or rec["video_id"])[:38], "acc") + chip(f'{esc(rec["lang_name"])} video', "ok")
        st.html(f'<div class="conv-bar"><span class="conv-title">{esc(conv["title"])}</span>{pill}</div>')
    with c2:
        st.selectbox("Answer language", ANSWER_LANG_OPTIONS, key="answer_lang", format_func=fmt_answer_compact,
                     label_visibility="collapsed", help="Language of the answers. Changing it never re-indexes the video.")
    with c3:
        with st.popover("", icon=":material/more_horiz:", width="stretch"):
            ids = index_ids()
            if ids:
                if conv["video_id"] in ids:
                    ss.src_select = conv["video_id"]
                st.selectbox("Knowledge source", ids, key="src_select", format_func=source_label,
                             on_change=cb_switch_source)
            st.text_input("Rename conversation", value=conv["title"], key=f"rn_{conv['id']}", max_chars=80)
            st.button("Save name", key=f"rnsave_{conv['id']}", on_click=cb_rename, args=(conv["id"],), width="stretch")
            st.button("Clear messages", key="clr", icon=":material/mop:", on_click=cb_clear, args=(conv["id"],),
                      disabled=not conv["messages"], width="stretch")
            st.button("Delete conversation", key="del", icon=":material/delete:", on_click=cb_delete,
                      args=(conv["id"],), width="stretch")


def view_chat():
    rec = store["indexes"].get(conv["video_id"]) if conv["video_id"] else None

    question = st.chat_input(
        "Ask in any language…" if rec else "Add a video first to start chatting",
        disabled=rec is None,
    )
    replace_last = False
    if ss.pending:
        question = ss.pending["q"]
        replace_last = ss.pending["regen"]
        ss.pending = None

    if ss.flash:
        st.toast(ss.flash, icon=":material/check_circle:")
        ss.flash = None

    if question and rec and not replace_last:
        conv["messages"].append({"id": uuid.uuid4().hex[:10], "role": "user", "content": question})
        if conv["title"] == DEFAULT_TITLE:
            conv["title"] = question.strip().replace("\n", " ")[:48]
        conv["updated"] = time.time()

    if not conv["messages"] and not question:
        st.html(
            '<div class="hero2"><span class="h-pill"><b></b>ask in any language · answer in the one you choose</span>'
            '<div class="h-title">Talk to any <span class="rot"><span>lecture</span><span>podcast</span>'
            '<span>keynote</span><span>tutorial</span></span></div>'
            '<div class="h-sub">Paste a YouTube link. The video and the answer can be in different languages.</div></div>'
            + ticker_html()
        )
        if rec:
            with st.container(key="langbox"):
                lang_bridge(rec)
            st.html('<div class="sug-cap"><span class="chip acc">try asking</span></div>')
            with st.container(key="sugs"):
                sc = st.columns(2)
                for i, text in enumerate(SUGGESTIONS):
                    with sc[i % 2]:
                        st.button(text, key=f"sug_{i}", icon=":material/arrow_outward:",
                                  on_click=cb_suggest, args=(text,), width="stretch")
            st.html('<div class="rib-note" style="margin-top:14px">Want another video or language? Open the Sources tab.</div>')
        else:
            onboard_card("Paste a YouTube link to begin")
        st.html(ribbon_html(idle_stages(), demo=True, still=False))
        return

    render_chat_bar(conv, rec)
    render_history(conv)
    if question and rec:
        stream_answer(conv, rec, question, replace_last)


# ============================================================
# SOURCES VIEW
# ============================================================

def view_sources():
    st.html('<div class="view-title">Sources</div><div class="view-sub">One index per video and caption language, reused '
            'every time. Each conversation uses exactly one source.</div>')
    onboard_card("Add a video")

    ids = index_ids()
    if not ids:
        st.html('<div class="panel" style="margin-top:20px"><h4>Nothing indexed yet</h4>'
                '<div class="sub">Paste a link above to build your first knowledge base.</div></div>')
    for key in ids:
        rec = store["indexes"][key]
        vid = rec["video_id"]
        active = conv["video_id"] == key
        t = rec["timings"]
        thumb = (f'<img src="https://img.youtube.com/vi/{esc(vid)}/mqdefault.jpg" alt="" '
                 f'onerror="this.style.visibility=\'hidden\'">')
        chips = "".join([
            chip(f'{esc(rec["lang_name"])} · {rec["lang_kind"]} captions', "ok"),
            chip(f'{rec["n_chunks"]:,} chunks', "acc"),
            chip(f'{rec["n_chars"]:,} chars'),
            chip(f'indexed in {fmt_s(rec["total_s"])}'),
        ])
        trk = "".join(chip(f'{esc(x["name"])} · {esc(x["code"])}' + (" · auto" if x["generated"] else ""),
                          "acc" if x["code"] == rec["lang_code"] else "") for x in rec["tracks"])
        st.html(
            f'<div class="panel" style="margin-top:14px"><div class="src">{thumb}<div>'
            f'<div class="src-t">{esc(rec.get("title") or vid)}{" " + chip("in use", "ok") if active else ""}</div>'
            f'<div class="src-id">{esc(vid)} · {esc(rec["lang_code"])}</div><div class="src-stats">{chips}</div>'
            f'<div class="src-stats" style="margin-top:8px">'
            f'{chip("fetch " + fmt_s(t["fetch"]))}{chip("split " + fmt_s(t["split"]))}'
            f'{chip("embed " + fmt_s(t["embed"]))}{chip("faiss " + fmt_s(t["faiss"]))}'
            f'</div></div></div>'
            f'<details class="pp"><summary>Caption languages on this video ({len(rec["tracks"])})</summary>'
            f'<div class="srcs" style="padding:10px 12px;margin:0">{trk}</div></details></div>'
        )
        b1, b2, _ = st.columns([1.5, 1.3, 2.2], gap="small")
        with b1:
            st.button("Chat with this" if not active else "Open chat", key=f"use_{key}",
                      type="primary" if not active else "secondary", icon=":material/forum:",
                      on_click=cb_use_source, args=(key,), width="stretch")
        with b2:
            st.button("Remove", key=f"rm_{key}", icon=":material/delete:",
                      on_click=cb_remove_source, args=(key,), width="stretch",
                      help="Free this index from memory (the transcript cache is kept)")


# ============================================================
# INSIGHTS VIEW (performance · models · preferences)
# ============================================================

def model_card(title, path, loaded, load_s, extra_rows):
    info = model_dir_info(path)
    status = chip("loaded", "ok") if loaded else (chip("not loaded") if info["exists"] else chip("missing", "err"))
    rows = [("Path", esc(path)),
            ("On disk", f'{info["size_gb"]:.2f} GB' if info["size_gb"] is not None else "—"),
            ("Load time", fmt_s(load_s) if load_s else "—")] + extra_rows
    kv = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    st.html(f'<div class="panel"><div style="display:flex;justify-content:space-between;align-items:center">'
            f'<h4>{esc(title)}</h4>{status}</div><dl class="kv" style="margin-top:12px">{kv}</dl></div>')
    return info


def tab_performance():
    perf = store["perf"]
    rec = store["indexes"].get(conv["video_id"]) if conv["video_id"] else None
    if not perf and not store["indexes"]:
        st.html('<div class="panel"><h4>No measurements yet</h4>'
                '<div class="sub">Index a video and ask a question — real timings show up here.</div></div>')
        return
    if perf:
        last = perf[-1]
        tiles = "".join([
            stat_tile("Retrieval", fmt_s(last["retrieval_s"]), "query embed + MMR"),
            stat_tile("Model load", fmt_s(last["model_load_s"]) if last["model_load_s"] else "cached",
                      "this query" if last["model_load_s"] else "already in memory"),
            stat_tile("First token", fmt_s(last["ttft_s"]), "time to first token", "acc"),
            stat_tile("Generation", fmt_s(last["gen_s"]), "total decode time"),
            stat_tile("Tokens", fmt_num(last["tokens"]), "generated"),
            stat_tile("Speed", f'{last["tps"]:.1f}' if last["tps"] else "—", "tokens / second", "ok"),
        ])
        st.html(f'<div class="tiles">{tiles}</div>')
        qmsg = next((m for m in reversed(conv["messages"]) if m.get("trace")), None)
        tr = qmsg["trace"] if qmsg else None
        if tr:
            decode = max((tr["gen_s"] or 0) - (tr["ttft_s"] or 0), 0)
            st.html('<div class="panel"><h4>Where the time went</h4>'
                    '<div class="sub">Latest answer in this conversation, end to end.</div>'
                    + stacked_bar_html([
                        ("Query translation", tr.get("bridge_s")), ("Query embedding", tr["qembed_s"]), ("MMR retrieval", tr["mmr_s"]),
                        ("Context build", tr["context_s"]), ("Model load", tr["model_load_s"]),
                        ("Prefill (to first token)", tr["ttft_s"]), ("Decode", decode),
                    ]) + '</div>')
    if rec:
        st.html(f'<div class="panel"><h4>Indexing · {esc(rec.get("title") or rec["video_id"])}</h4>'
                f'<div class="sub">{rec["n_chunks"]:,} chunks · {rec["n_chars"]:,} characters</div>'
                + stacked_bar_html([
                    ("Transcript fetch", rec["timings"]["fetch"]), ("Splitting", rec["timings"]["split"]),
                    ("Embedding", rec["timings"]["embed"]), ("FAISS build", rec["timings"]["faiss"]),
                ]) + '</div>')
    if perf:
        tps_vals = [p["tps"] for p in perf if p["tps"]]
        max_tps = max(tps_vals) if tps_vals else 1
        rows = []
        for i, p in enumerate(reversed(perf[-10:]), start=1):
            n = len(perf) - i + 1
            w = (p["tps"] / max_tps * 70) if p["tps"] else 0
            speed = f'<i class="bar" style="width:{w:.0f}px"></i>{p["tps"]:.1f}' if p["tps"] else "—"
            rows.append(f'<tr><td>{n}</td><td>{esc(p["video_id"])}</td><td>{fmt_s(p["retrieval_s"])}</td>'
                        f'<td>{fmt_s(p["ttft_s"])}</td><td>{fmt_s(p["gen_s"])}</td><td>{fmt_num(p["tokens"])}</td><td>{speed}</td></tr>')
        st.html('<div class="panel"><h4>Query history</h4><div class="sub">Most recent 10 this session.</div>'
                '<table class="hist"><tr><th>#</th><th>Video</th><th>Retrieval</th><th>TTFT</th>'
                '<th>Generation</th><th>Tokens</th><th>Tok/s</th></tr>' + "".join(rows) + '</table></div>')


def tab_models():
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ecfg = model_dir_info(EMBEDDING_MODEL_PATH)["config"]
        model_card("Embedding · Qwen3-Embedding-0.6B", EMBEDDING_MODEL_PATH, "embed_load_s" in store,
                   store.get("embed_load_s"),
                   [("Normalization", "normalize_embeddings=True"), ("Hidden size", str(ecfg.get("hidden_size", "—")))])
    with c2:
        lcfg = model_dir_info(LLM_MODEL_PATH)["config"]
        q = lcfg.get("quantization") or {}
        model_card("LLM · Qwen3-8B 4-bit (MLX)", LLM_MODEL_PATH, "llm_load_s" in store, store.get("llm_load_s"),
                   [("Quantization", f'{q.get("bits", "—")}-bit · group {q.get("group_size", "—")}'),
                    ("Sampler", f"temperature {TEMPERATURE}"),
                    ("Streaming", "stream_generate" if STREAMING_AVAILABLE else "generate (no streaming)")])
        a, b, _ = st.columns([1.3, 1.3, 1.4], gap="small")
        with a:
            if st.button("Load now", icon=":material/play_arrow:", disabled="llm_load_s" in store, width="stretch"):
                try:
                    with st.spinner("Loading Qwen3-8B…"):
                        _, _, secs = get_llm_timed()
                    st.toast(f"LLM loaded in {fmt_s(secs)}", icon=":material/check_circle:")
                    st.rerun()
                except Exception as exc:
                    log.exception("LLM load failed")
                    st.error(friendly_error(exc))
        with b:
            st.button("Unload", icon=":material/eject:", on_click=cb_unload,
                      disabled="llm_load_s" not in store, width="stretch", help="Release the model from memory")
    st.html(
        '<div class="panel"><h4>Retrieval (locked)</h4><div class="sub">Kept exactly as in your original pipeline.</div><dl class="kv">'
        f'<dt>Splitter</dt><dd>RecursiveCharacterTextSplitter · {CHUNK_SIZE} / {CHUNK_OVERLAP}</dd>'
        f'<dt>Search</dt><dd>FAISS · MMR · k={RETRIEVE_K} · fetch_k={FETCH_K} · lambda_mult={LAMBDA_MULT}</dd>'
        '<dt>Grounding</dt><dd>only the retrieved context, otherwise "I don\'t know."</dd></dl></div>'
    )
    st.select_slider("Maximum response length (tokens)", options=[256, 512, 768, 1024, 1536, 2048], key="max_tokens")
    st.caption("Qwen3 may spend part of this budget reasoning before it answers. If answers get cut off, raise it.")


def tab_prefs():
    st.toggle("Boost retrieval for Telugu, Tamil, Kannada, Malayalam, Bengali, Gujarati, Punjabi and Arabic-script questions",
              key="bridge_boost",
              help="Qwen3-8B translates the question to English for the passage search only (about 1.5 s). "
                   "The answer is still written for your original question.")
    st.toggle("Show model reasoning (Qwen3 thinking)", key="show_reasoning",
              help="Display only. Generation itself is unchanged.")
    st.html(
        '<div class="panel"><h4>What is and is not supported</h4><div class="sub">Video language and answer language are independent.</div>'
        '<dl class="kv tight" style="grid-template-columns:150px 1fr">'
        '<dt>Video languages</dt><dd>Only caption tracks YouTube really has for that video (manual or auto-generated). '
        'Use the Captions button to see them. This app does not machine-translate captions.</dd>'
        '<dt>Answer languages</dt><dd>20 offered. Qwen3-8B 4-bit writes the answer; quality is best in widely used languages '
        'and can be weaker in low-resource ones (for example Telugu or Tamil). Check important answers.</dd>'
        '<dt>Embedding model</dt><dd>Qwen3-Embedding-0.6B (unchanged). Its model card lists 100+ languages and cross-lingual retrieval.</dd>'
        '<dt>Measured here</dt><dd>4 questions about one video, asked in 6 languages, against English, Japanese and Spanish transcripts. '
        'Hindi, Japanese, Spanish and English questions found the same passage as the English baseline in 4/4 cases. '
        'Telugu found it in 3/4 and Tamil in 2/4 directly, and 4/4 with the retrieval boost. Small sample, not a benchmark.</dd>'
        '</dl></div>'
    )
    st.caption("Chats and indexes live in this server process: they survive page reloads, not a server restart.")
    c1, c2, _ = st.columns([1.5, 1.5, 2], gap="small")
    with c1:
        st.button("Delete all chats", icon=":material/delete_sweep:", on_click=cb_wipe_all, width="stretch")
    with c2:
        if st.button("Clear transcript cache", icon=":material/cleaning_services:", width="stretch"):
            load_transcript.clear()
            list_transcript_tracks.clear()
            store["transcripts_seen"].clear()
            st.toast("Transcript cache cleared.", icon=":material/check_circle:")
    rows = []
    for pkg in ["streamlit", "mlx", "mlx-lm", "langchain-core", "langchain-community",
                "langchain-huggingface", "faiss-cpu", "sentence-transformers", "youtube-transcript-api"]:
        try:
            ver = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            ver = "not installed"
        rows.append(f"<dt>{pkg}</dt><dd>{ver}</dd>")
    st.html(f'<div class="panel" style="margin-top:14px"><h4>Diagnostics</h4><dl class="kv" style="margin-top:10px">{"".join(rows)}'
            f'<dt>Streaming</dt><dd>{"available" if STREAMING_AVAILABLE else "unavailable"}</dd></dl></div>')


def view_insights():
    st.html('<div class="view-title">Insights</div><div class="view-sub">Real measurements from this machine. No estimates.</div>')
    t1, t2, t3 = st.tabs(["Performance", "Models", "Preferences"])
    with t1:
        tab_performance()
    with t2:
        tab_models()
    with t3:
        tab_prefs()


# ============================================================
# ROUTER
# ============================================================

try:
    {"Chat": view_chat, "Sources": view_sources, "Insights": view_insights}[VIEW]()
except Exception as exc:  # last-resort guard so the shell never white-screens
    log.exception("Unhandled UI error")
    st.error(f"Unexpected error: {friendly_error(exc)}")
