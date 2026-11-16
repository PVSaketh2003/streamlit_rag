from . import rag_core as core


def source_dict(s):
    return {
        "id": s.id, "key": s.key, "video_id": s.video_id, "title": s.title or "",
        "lang_code": s.lang_code, "lang_name": s.lang_name, "lang_kind": s.lang_kind,
        "lang_reason": s.lang_reason, "tracks": s.tracks, "n_chunks": s.n_chunks, "n_chars": s.n_chars,
        "n_words": s.n_words, "timings": s.timings, "details": s.details, "total_s": s.total_s,
        "built_at": s.built_at.isoformat(),
        "thumbnail": f"https://img.youtube.com/vi/{s.video_id}/mqdefault.jpg",
    }


def message_dict(m):
    return {
        "id": m.id, "role": m.role, "content": m.content, "reasoning": m.reasoning,
        "error": m.error, "trace": m.trace, "created": m.created.isoformat(),
    }


def conversation_dict(c, with_messages=False, message_count=None):
    d = {
        "id": str(c.id), "title": c.title, "source_id": c.source_id,
        "created": c.created.isoformat(), "updated": c.updated.isoformat(),
    }
    if message_count is not None:
        d["message_count"] = message_count
    if with_messages:
        d["messages"] = [message_dict(m) for m in c.messages.all()]
    return d


def settings_dict(s):
    return {"max_tokens": s.max_tokens, "answer_lang": s.answer_lang, "video_lang": s.video_lang,
            "bridge_boost": s.bridge_boost, "show_reasoning": s.show_reasoning}


def metric_dict(m):
    return {"id": m.id, "ts": m.created.isoformat(), "video_id": m.video_id, "retrieval_s": m.retrieval_s,
            "model_load_s": m.model_load_s, "ttft_s": m.ttft_s, "gen_s": m.gen_s, "tokens": m.tokens,
            "tps": m.tps, "total_s": m.total_s}


def langs_payload():
    return [{"code": c, "name": n, "native": nat, "script": sc} for c, n, nat, sc in core.LANGS]
