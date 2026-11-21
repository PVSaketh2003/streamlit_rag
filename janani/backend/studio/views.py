import importlib.metadata as importlib_metadata
import logging
import time

from django.db import connection, transaction
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import rag_core as core
from .api_errors import ApiError
from .models import AppSettings, Chunk, Conversation, Message, QueryMetric, UserSource, VideoSource
from .serializers import (conversation_dict, langs_payload, message_dict, metric_dict, settings_dict,
                          source_dict)
from .sse import sse_event, sse_response

log = logging.getLogger("studio.api")

DEFAULT_TITLE = "New conversation"
MAX_TOKEN_OPTIONS = [256, 512, 768, 1024, 1536, 2048]
MAX_QUESTION_CHARS = 4000
SUGGESTIONS = ["Summarize the main ideas", "Explain the key concepts",
               "What examples are discussed?", "What does the speaker conclude?"]
STAGES = [
    {"key": "fetch", "label": "Transcript fetch", "kind": "idx"},
    {"key": "split", "label": "Text splitting", "kind": "idx"},
    {"key": "embed", "label": "Embedding generation", "kind": "idx"},
    {"key": "faiss", "label": "FAISS indexing", "kind": "idx"},
    {"key": "qembed", "label": "Query embedding", "kind": "qry"},
    {"key": "mmr", "label": "MMR retrieval", "kind": "qry"},
    {"key": "context", "label": "Context construction", "kind": "qry"},
    {"key": "infer", "label": "MLX inference", "kind": "qry"},
    {"key": "answer", "label": "Final answer", "kind": "qry"},
]


def _body(request):
    data = request.data
    if not isinstance(data, dict):
        raise ApiError("Expected a JSON object.")
    return data


def _link_source(account, source):
    UserSource.objects.get_or_create(account=account, source=source)


def _my_sources(account):
    return VideoSource.objects.filter(user_links__account=account)


def _get_or_404(model, **kw):
    try:
        return model.objects.get(**kw)
    except (model.DoesNotExist, ValueError):
        raise ApiError(f"{model.__name__} not found.", status=404, code="not_found") from None


# ------------------------------------------------------------------ meta

def spa(request):
    """Serve the built React app's index.html for client-side routes."""
    from django.conf import settings
    from django.http import HttpResponse
    index = settings.FRONTEND_DIST / "index.html"
    if not index.is_file():
        return HttpResponse("Frontend not built. Run `npm run build` in janani/frontend (or use the Vite dev server on :5173).",
                            status=404, content_type="text/plain")
    resp = HttpResponse(index.read_bytes(), content_type="text/html; charset=utf-8")
    resp["Cache-Control"] = "no-cache"
    return resp


@api_view(["GET"])
def health(request):
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
        db = True
    except Exception:
        db = False
    from django.conf import settings
    from . import mailer
    return Response({"ok": db, "database": db, "version": settings.APP_VERSION, "runtime": core.runtime_status(),
                     "email": mailer.status()},
                    status=200 if db else 503)


@api_view(["GET"])
def config(request):
    return Response({
        "langs": langs_payload(),
        "stages": STAGES,
        "suggestions": SUGGESTIONS,
        "max_token_options": MAX_TOKEN_OPTIONS,
        "default_video": core.DEFAULT_VIDEO,
        "low_resource_scripts": sorted(core.LOW_RESOURCE_SCRIPTS),
        "retrieval": {"chunk_size": core.CHUNK_SIZE, "chunk_overlap": core.CHUNK_OVERLAP, "k": core.RETRIEVE_K,
                      "fetch_k": core.FETCH_K, "lambda_mult": core.LAMBDA_MULT, "temperature": core.TEMPERATURE},
        "settings": settings_dict(AppSettings.load(request.account)),
        "account": {"email": request.account.email},
        "youtube": {"yt_dlp": core.yt_dlp_available(), "proxy": bool(core.YT_PROXY_URL)},
        "runtime": core.runtime_status(),
    })


@api_view(["GET", "PATCH"])
def app_settings(request):
    s = AppSettings.load(request.account)
    if request.method == "PATCH":
        d = _body(request)
        if "max_tokens" in d:
            if d["max_tokens"] not in MAX_TOKEN_OPTIONS:
                raise ApiError(f"max_tokens must be one of {MAX_TOKEN_OPTIONS}.")
            s.max_tokens = d["max_tokens"]
        if "answer_lang" in d:
            if d["answer_lang"] not in core.LANG_BY_CODE:
                raise ApiError("Unknown answer language.")
            s.answer_lang = d["answer_lang"]
        if "video_lang" in d:
            if d["video_lang"] not in ["auto", "other", *core.LANG_BY_CODE]:
                raise ApiError("Unknown video language.")
            s.video_lang = d["video_lang"]
        for flag in ("bridge_boost", "show_reasoning"):
            if flag in d:
                if not isinstance(d[flag], bool):
                    raise ApiError(f"{flag} must be true or false.")
                setattr(s, flag, d[flag])
        s.save()
    return Response(settings_dict(s))


# ------------------------------------------------------------------ video + sources

def _resolve_video(d):
    try:
        return core.extract_youtube_video_id(d.get("url"))
    except ValueError as exc:
        raise ApiError(str(exc), code="invalid_link") from None


def _tracks(video_id):
    try:
        return core.list_transcript_tracks(video_id)
    except core.YouTubeBlocked as exc:
        raise ApiError(str(exc), status=503, code="youtube_blocked",
                       extra={"yt_dlp": core.yt_dlp_available(), "proxy": bool(core.YT_PROXY_URL)}) from None
    except Exception as exc:
        log.warning("Caption listing failed for %s: %s", video_id, type(exc).__name__)
        raise ApiError(core.friendly_error(exc), status=422, code="captions_unavailable") from None


@api_view(["POST"])
def inspect_video(request):
    d = _body(request)
    video_id = _resolve_video(d)
    blocked = False
    try:
        tracks = _tracks(video_id)
    except ApiError as exc:
        if exc.code != "youtube_blocked":
            raise
        tracks, blocked = [], True      # still show title/thumbnail; indexing falls back to Gemini / pasted text
    return Response({
        "video_id": video_id, "title": core.fetch_video_title(video_id) or "", "tracks": tracks, "blocked": blocked,
        "fallback": "gemini" if blocked and core.gemini_fallback_available() else None,
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        "existing": [source_dict(s) for s in _my_sources(request.account).filter(video_id=video_id)],
    })


@api_view(["GET"])
def sources(request):
    return Response({"results": [source_dict(s) for s in _my_sources(request.account)],
                     "building": sorted(core.building_keys())})


@api_view(["DELETE"])
def source_detail(request, pk):
    s = _get_or_404(VideoSource, pk=pk, user_links__account=request.account)
    UserSource.objects.filter(account=request.account, source=s).delete()
    Conversation.objects.filter(account=request.account, source=s).update(source=None)
    if not UserSource.objects.filter(source=s).exists():          # last person to remove it frees the shared index
        index_file = s.index_file
        s.delete()
        core.drop_index(pk, index_file)
    return Response(status=204)


@api_view(["POST"])
def index_source(request):
    """Index (or reuse) one video. Streams pipeline stages as Server-Sent Events."""
    d = _body(request)
    video_id = _resolve_video(d)
    requested = d.get("video_lang") or AppSettings.load(request.account).video_lang
    try:
        tracks = _tracks(video_id)
    except ApiError as exc:
        if exc.code == "youtube_blocked" and core.gemini_fallback_available():
            return sse_response(_gemini_stream(video_id, request.account))     # YouTube blocks us: Gemini listens
        raise
    try:
        track, reason = core.pick_track(tracks, requested)
    except core.LanguageUnavailable as exc:
        raise ApiError(str(exc), status=409, code="language_unavailable",
                       extra={"tracks": tracks, "requested": requested}) from None
    return sse_response(_build_stream(video_id, track, reason, tracks, account=request.account))


def _gemini_stream(video_id, account):
    """Blocked-by-YouTube path: Gemini transcribes the audio (chunked, with live progress), then the normal
    split -> embed -> FAISS pipeline runs on that text."""
    from .models import TranscriptCache

    def stream():
        reason = "YouTube blocks caption downloads from this server, so Gemini transcribed the video's audio"
        yield sse_event("stage", {"key": "fetch", "status": "running", "ms": None,
                                  "detail": "asking Gemini to transcribe the audio"})
        text, lang, minutes, partial, gap_min = None, None, 0, False, 0
        cached = TranscriptCache.objects.filter(video_id=video_id, origin="gemini").first()
        t0 = time.perf_counter()
        try:
            if cached:
                text, lang = cached.text, cached.lang_code
            else:
                for ev in core.gemini_transcribe(video_id):
                    if ev[0] == "progress":
                        yield sse_event("stage", {"key": "fetch", "status": "running", "ms": None, "detail": ev[2]})
                    elif ev[0] == "ping":
                        yield ": keep-alive\n\n"
                    elif ev[0] == "done":
                        _, text, lang, minutes, partial, gap_min = ev
                core.save_transcript(video_id, lang, text, "gemini")
        except GeneratorExit:
            raise
        except Exception as exc:
            log.warning("Gemini transcription failed for %s: %s", video_id, exc)
            yield sse_event("stage", {"key": "fetch", "status": "error", "ms": None, "detail": "transcription failed"})
            yield sse_event("error", {"message": core.friendly_error(exc), "code": "youtube_blocked"})
            return
        info = core.lang_info(lang)
        track = {"code": lang, "name": info["name"], "generated": True, "translatable": False}
        label = "transcribed from audio by Gemini"
        if partial:
            label += f" (first {minutes} min)"
        if not cached and gap_min:
            label += f" · {gap_min} min could not be transcribed"
        yield sse_event("meta", {"video_id": video_id, "track": track, "reason": reason, "reused": False})
        yield from _build_stream(video_id, track, reason, [track], transcript_text=text, origin_label=label,
                                 emit_meta=False, account=account)

    return stream()


def _build_stream(video_id, track, reason, tracks, transcript_text=None, origin_label=None, emit_meta=True, account=None):
    key = core.index_key(video_id, track["code"])
    existing = VideoSource.objects.filter(key=key).first()
    if existing and existing.embed_model != core.EMBED_TAG:
        core.drop_index(existing.id, existing.index_file)    # stale embedding model: rebuild transparently
        existing.delete()
        existing = None

    def stream():
        if emit_meta:
            yield sse_event("meta", {"video_id": video_id, "track": track, "reason": reason, "reused": bool(existing)})
        if existing:
            _link_source(account, existing)
            yield sse_event("done", {"source": source_dict(existing), "reused": True})
            return
        try:
            for ev in core.build_index(video_id, track, reason, tracks, transcript_text=transcript_text,
                                       origin_label=origin_label):
                if ev[0] == "stage":
                    _, skey, status, ms, detail = ev
                    yield sse_event("stage", {"key": skey, "status": status, "ms": ms, "detail": detail})
                elif ev[0] == "done":
                    data = ev[1]
                    vs, emb = data.pop("_vs"), data.pop("_emb")
                    texts = data.pop("chunk_texts")
                    data.pop("key")
                    with transaction.atomic():
                        src = VideoSource.objects.create(key=key, **data)
                        Chunk.objects.bulk_create([Chunk(source=src, idx=i, text=t) for i, t in enumerate(texts)],
                                                  batch_size=500)
                    core.register_index(src.id, vs, emb)
                    _link_source(account, src)
                    log.info("Indexed %s (%s): %d chunks, %.2fs", video_id, track["code"], src.n_chunks, src.total_s)
                    yield sse_event("done", {"source": source_dict(src), "reused": False})
        except GeneratorExit:
            raise
        except Exception as exc:
            log.warning("Indexing failed for %s: %s", video_id, exc)
            code = "youtube_blocked" if isinstance(exc, core.YouTubeBlocked) else "index_failed"
            yield sse_event("error", {"message": core.friendly_error(exc), "code": code})

    return stream()


MAX_PASTE_CHARS = 2_000_000


@api_view(["POST"])
def index_manual(request):
    """Index a transcript the user pasted (works even when YouTube blocks caption downloads)."""
    d = _body(request)
    video_id = _resolve_video(d)
    text = str(d.get("text") or "")
    # strip timestamps such as 0:12, 1:02:33 or [00:12] that come with YouTube's "Show transcript" panel
    text = core.re.sub(r"\[?\b\d{1,2}:\d{2}(?::\d{2})?\]?", " ", text)
    text = core.re.sub(r"\s+", " ", text).strip()
    if len(text) < 80:
        raise ApiError("Paste the full transcript (at least a few sentences).")
    if len(text) > MAX_PASTE_CHARS:
        raise ApiError("That transcript is too long (2 million characters max).")
    lang = str(d.get("lang_code") or "en").strip()[:16]
    if not core.re.fullmatch(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?", lang):
        raise ApiError("Pick the language the transcript is written in.")
    info = core.lang_info(lang)
    track = {"code": lang, "name": info["name"], "generated": False, "translatable": False}
    core.save_transcript(video_id, lang, text, "manual")
    return sse_response(_build_stream(video_id, track, "transcript pasted by you", [track], transcript_text=text,
                                      account=request.account))


# ------------------------------------------------------------------ conversations

@api_view(["GET", "POST", "DELETE"])
def conversations(request):
    if request.method == "GET":
        from django.db.models import Count
        qs = Conversation.objects.filter(account=request.account).annotate(n=Count("messages")).filter(n__gt=0)[:100]
        return Response({"results": [conversation_dict(c, message_count=c.n) for c in qs]})
    if request.method == "DELETE":
        Conversation.objects.filter(account=request.account).delete()
        QueryMetric.objects.filter(account=request.account).delete()
        return Response(status=204)
    d = _body(request)
    source = None
    if d.get("source_id") is not None:
        source = _get_or_404(VideoSource, pk=d["source_id"], user_links__account=request.account)
    # reuse a pristine empty conversation instead of piling up blanks
    blank = Conversation.objects.filter(account=request.account, messages__isnull=True).first()
    if blank:
        blank.source = source or blank.source
        blank.save()
        return Response(conversation_dict(blank, with_messages=True), status=200)
    c = Conversation.objects.create(account=request.account, source=source or _my_sources(request.account).first())
    return Response(conversation_dict(c, with_messages=True), status=201)


@api_view(["GET", "PATCH", "DELETE"])
def conversation_detail(request, pk):
    c = _get_or_404(Conversation, pk=pk, account=request.account)
    if request.method == "DELETE":
        c.delete()
        return Response(status=204)
    if request.method == "PATCH":
        d = _body(request)
        if "title" in d:
            title = str(d["title"]).strip()[:80]
            if not title:
                raise ApiError("Title cannot be empty.")
            c.title = title
        if "source_id" in d:
            if c.messages.exists() and d["source_id"] != c.source_id:
                raise ApiError("A conversation with messages keeps its source. Start a new chat for another video.",
                               status=409, code="source_locked")
            c.source = (_get_or_404(VideoSource, pk=d["source_id"], user_links__account=request.account)
                        if d["source_id"] is not None else None)
        c.save()
    return Response(conversation_dict(c, with_messages=True))


@api_view(["POST"])
def conversation_clear(request, pk):
    c = _get_or_404(Conversation, pk=pk, account=request.account)
    c.messages.all().delete()
    c.title = DEFAULT_TITLE
    c.save()
    return Response(conversation_dict(c, with_messages=True))


@api_view(["POST"])
def ask(request, pk):
    """One RAG turn, streamed as SSE: start → stage* → token* → done | error."""
    conv = _get_or_404(Conversation, pk=pk, account=request.account)
    d = _body(request)
    if not conv.source_id:
        raise ApiError("Add a video first, then ask your question.", code="no_source")
    source = conv.source
    cfg = AppSettings.load(request.account)
    account = request.account
    answer_lang = d.get("answer_lang") or cfg.answer_lang
    if answer_lang not in core.LANG_BY_CODE:
        raise ApiError("Unknown answer language.")
    max_tokens = cfg.max_tokens
    regenerate = bool(d.get("regenerate"))

    if regenerate:
        trailing = list(conv.messages.order_by("-id"))
        while trailing and trailing[0].role == "assistant":
            trailing.pop(0).delete()
        last_user = trailing[0] if trailing else None
        if not last_user:
            raise ApiError("There is nothing to regenerate.")
        question, user_msg = last_user.content, last_user
    else:
        question = str(d.get("question") or "").strip()
        if not question:
            raise ApiError("Type a question first.")
        if len(question) > MAX_QUESTION_CHARS:
            raise ApiError(f"Questions are limited to {MAX_QUESTION_CHARS:,} characters.")
        user_msg = Message.objects.create(conversation=conv, role="user", content=question)
        if conv.title == DEFAULT_TITLE:
            conv.title = question.replace("\n", " ")[:48]
        conv.save()

    def stream():
        raw, trace = "", None
        stages = {k: {"status": "idle", "ms": None, "detail": ""} for k in core.QUERY_STAGES}
        finished = False
        yield sse_event("start", {"user_message": message_dict(user_msg), "title": conv.title})
        try:
            for ev in core.run_query(source, question, max_tokens, answer_lang, cfg.bridge_boost):
                kind = ev[0]
                if kind == "stage":
                    _, key, status, ms, detail = ev
                    stages[key] = {"status": status, "ms": ms, "detail": detail}
                    yield sse_event("stage", {"key": key, "status": status, "ms": ms, "detail": detail})
                elif kind == "token":
                    raw += ev[1]
                    yield sse_event("token", {"t": ev[1]})
                elif kind == "ping":
                    yield ": keep-alive\n\n"
                elif kind == "done":
                    trace = ev[1]
            reasoning, answer, thinking = core.split_reasoning(raw)
            note = ""
            if thinking and not answer:
                note = ("The model used its entire token budget on reasoning before answering. "
                        "Raise the maximum response length in Insights → Models and retry.")
            elif not answer:
                note = "The model returned an empty response."
            if note:
                trace["finish_reason"] = "length"
            trace["stages"] = stages
            msg = Message.objects.create(conversation=conv, role="assistant", content=answer or note,
                                         reasoning=reasoning, trace=trace)
            QueryMetric.objects.create(
                account=account, video_id=source.video_id, retrieval_s=trace["bridge_s"] + trace["qembed_s"] + trace["mmr_s"],
                model_load_s=trace["model_load_s"], ttft_s=trace["ttft_s"], gen_s=trace["gen_s"],
                tokens=trace["tokens"], tps=trace["tps"], total_s=trace["total_s"])
            conv.save()
            finished = True
            yield sse_event("done", {"message": message_dict(msg)})
        except GeneratorExit:
            # Client pressed Stop / closed the tab: keep whatever was generated.
            if not finished:
                reasoning, answer, _ = core.split_reasoning(raw)
                if answer or reasoning:
                    Message.objects.create(conversation=conv, role="assistant", content=answer,
                                           reasoning=reasoning, trace={"stopped": True, "stages": stages,
                                                                       "sources": [], "prompt": ""})
                    conv.save()
            raise
        except Exception as exc:
            log.exception("Query failed")
            text = core.friendly_error(exc)
            msg = Message.objects.create(conversation=conv, role="assistant", error=text)
            conv.save()
            yield sse_event("error", {"message": text, "assistant_message": message_dict(msg)})

    return sse_response(stream())


# ------------------------------------------------------------------ insights / models

@api_view(["GET"])
def insights(request):
    mine = QueryMetric.objects.filter(account=request.account)
    history = [metric_dict(m) for m in mine[:10]]
    out = {"last": history[0] if history else None, "history": history,
           "total_queries": mine.count(), "latest_trace": None, "indexing": None}
    cid = request.query_params.get("conversation")
    source = None
    if cid:
        try:
            conv = Conversation.objects.filter(pk=cid, account=request.account).first()
        except ValueError:
            conv = None
        if conv:
            source = conv.source
            msg = conv.messages.filter(role="assistant", trace__isnull=False).exclude(trace__stopped=True).last()
            if msg:
                t = msg.trace
                out["latest_trace"] = {k: t.get(k) for k in (
                    "bridge_s", "qembed_s", "mmr_s", "context_s", "model_load_s", "ttft_s", "gen_s")}
    out["indexing"] = source_dict(source) if source else None
    return Response(out)


def _mask(key):
    return f"{key[:4]}…{key[-3:]}" if key and len(key) > 10 else ("set" if key else "")


@api_view(["GET"])
def models_info(request):
    rt = core.runtime_status()
    return Response({
        "embedding": {"provider": "Google Gemini", "model": core.GEMINI_EMBED_MODEL, "dim": core.EMBED_DIM,
                      "key_set": rt["embedding_ready"], "key_hint": _mask(core.GEMINI_API_KEY),
                      "verified": rt["embedding_verified"], "env": "GEMINI_API_KEY"},
        "llm": {"provider": core.LLM["provider"], "model": core.GROK_MODEL, "base_url": core.XAI_BASE_URL,
                "temperature": core.TEMPERATURE, "key_set": rt["llm_ready"], "key_hint": _mask(core.XAI_API_KEY),
                "verified": rt["llm_verified"], "env": core.LLM["env"]},
        "runtime": rt,
    })


def _check(fn):
    try:
        secs = fn()
    except Exception as exc:
        raise ApiError(core.friendly_error(exc), status=502, code="provider_error") from None
    return Response({"seconds": secs, "runtime": core.runtime_status()})


@api_view(["POST"])
def llm_check(request):
    return _check(core.check_llm)


@api_view(["POST"])
def embed_check(request):
    return _check(core.check_embeddings)


@api_view(["POST"])
def cache_clear(request):
    core.fetch_video_title.cache_clear()          # transcripts are shared and costly to rebuild: users cannot wipe them
    return Response({"ok": True})


@api_view(["GET"])
def diagnostics(request):
    pkgs = ["django", "djangorestframework", "psycopg", "openai", "google-genai", "langchain-core", "langchain-community",
            "faiss-cpu", "youtube-transcript-api", "yt-dlp"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            versions[p] = None
    with connection.cursor() as cur:
        cur.execute("SHOW server_version")
        pg = cur.fetchone()[0]
    return Response({"packages": versions, "postgres": pg, "streaming": True,
                     "faiss_dir": str(core.FAISS_DIR), "time": time.time()})
