"""Backend test suite (run: `python manage.py test studio`). External services (YouTube, Gemini, Groq/xAI)
are mocked, so the suite is deterministic and needs only PostgreSQL."""
import hashlib
import json
import time
from unittest import mock

import numpy as np
from django.test import Client, TestCase, override_settings

from . import rag_core as core
from .models import (Account, AuthSession, Conversation, LoginChallenge, Message, QueryMetric, TranscriptCache,
                     UserSource, VideoSource)
from .vstore import VectorIndex, maximal_marginal_relevance

DIM = 32
PASSWORD = "correct-horse-9"


def make_account(email="owner@example.com", password=PASSWORD):
    from django.contrib.auth.hashers import make_password
    return Account.objects.create(email=email, password_hash=make_password(password))


def login_client(account=None):
    """A test client that already carries a valid session cookie for `account`."""
    import secrets
    from datetime import timedelta
    from django.utils import timezone
    from .auth import SESSION_COOKIE, _hash_token
    account = account or make_account()
    token = secrets.token_urlsafe(32)
    AuthSession.objects.create(account=account, token_hash=_hash_token(token),
                               expires_at=timezone.now() + timedelta(days=1))
    c = Client()
    c.cookies[SESSION_COOKIE] = token
    c.account = account
    return c


def fake_vec(text):
    """Deterministic unit vector per text (same text -> same vector)."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()


def fake_embed_batch(texts, task):
    return [fake_vec(t) for t in texts]


def sse_events(response):
    body = b"".join(response.streaming_content).decode()
    events = []
    for block in body.split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if name:
            events.append((name, data))
    return events


TRANSCRIPT = (
    "LangChain chains connect several steps into one pipeline. A sequential chain runs the steps one after "
    "another and passes each output to the next step. A parallel chain runs several steps at the same time on "
    "the same input and merges the results. RunnableParallel is used to build parallel chains. "
) * 12


# ------------------------------------------------------------------ pure logic

class VideoIdTests(TestCase):
    def test_accepts_common_link_shapes(self):
        vid = "Ilg3gGewQ5U"
        for url in [vid, f"https://www.youtube.com/watch?v={vid}", f"youtube.com/watch?v={vid}&list=PL1&index=3",
                    f"https://youtu.be/{vid}?si=abc", f"https://www.youtube.com/shorts/{vid}",
                    f"https://www.youtube.com/embed/{vid}", f"https://m.youtube.com/watch?v={vid}&t=30s"]:
            self.assertEqual(core.extract_youtube_video_id(url), vid, url)

    def test_rejects_bad_input(self):
        for bad in ["", "https://vimeo.com/123", "https://www.youtube.com/playlist?list=PL1",
                    "https://www.youtube.com/watch?v=short", "not a link"]:
            with self.assertRaises(ValueError, msg=bad):
                core.extract_youtube_video_id(bad)


class LanguageTests(TestCase):
    TRACKS = [
        {"code": "hi", "name": "Hindi", "generated": True, "translatable": True},
        {"code": "en", "name": "English", "generated": False, "translatable": True},
        {"code": "en", "name": "English", "generated": True, "translatable": True},
    ]

    def test_pick_track_auto_uses_spoken_language(self):
        track, reason = core.pick_track(self.TRACKS, "auto")
        self.assertEqual(track["code"], "hi")
        self.assertIn("detected", reason)

    def test_pick_track_prefers_manual_captions(self):
        track, _ = core.pick_track(self.TRACKS, "en")
        self.assertFalse(track["generated"])

    def test_pick_track_unavailable_language(self):
        with self.assertRaises(core.LanguageUnavailable):
            core.pick_track(self.TRACKS, "ta")

    def test_detect_script(self):
        self.assertEqual(core.detect_script("What is a chain?"), "Latin")
        self.assertEqual(core.detect_script("సీక్వెన్షియల్ చైన్ అంటే ఏమిటి?"), "Telugu")
        self.assertEqual(core.detect_script("चेन क्या है"), "Devanagari")
        self.assertEqual(core.detect_script("チェーンとは"), "Japanese")

    def test_language_rules_keep_grounding_instruction(self):
        rules = core.language_rules("hi", "English")
        self.assertIn("Hindi (हिन्दी)", rules)
        self.assertIn("I don't know.", rules)
        prompt = core.format_prompt("CTX", "Q?", rules)
        self.assertIn("ONLY using the provided transcript context", prompt)
        self.assertIn("CTX", prompt)

    def test_split_reasoning(self):
        self.assertEqual(core.split_reasoning("<think>plan</think>Answer"), ("plan", "Answer", False))
        self.assertEqual(core.split_reasoning("<think>still thinking"), ("still thinking", "", True))
        self.assertEqual(core.split_reasoning("Plain"), ("", "Plain", False))


class ProviderErrorTests(TestCase):
    def _err(self, body, payload):
        import httpx
        from openai import BadRequestError
        resp = httpx.Response(400, request=httpx.Request("POST", "https://api.example/v1"), json=payload)
        return BadRequestError("Error code: 400", response=resp, body=body)

    def test_reads_xai_style_string_body(self):
        e = self._err("Incorrect API key provided.", {"code": "invalid-argument", "error": "Incorrect API key provided."})
        self.assertEqual(core._provider_message(e), "Incorrect API key provided.")
        self.assertIn("rejected the API key", core.friendly_error(e))

    def test_reads_openai_style_dict_body(self):
        e = self._err({"message": "Unsupported parameter: stream_options"}, {"error": {"message": "x"}})
        self.assertEqual(core._provider_message(e), "Unsupported parameter: stream_options")

    def test_falls_back_to_raw_response(self):
        e = self._err(None, {"error": {"message": "model gone"}})
        self.assertEqual(core._provider_message(e), "model gone")

    def test_fatal_errors_are_not_retried(self):
        self.assertTrue(core._is_fatal("Incorrect API key provided"))
        self.assertTrue(core._is_fatal("The model `x` does not exist"))
        self.assertFalse(core._is_fatal("Unsupported parameter: stream_options"))


# ------------------------------------------------------------------ retrieval accuracy

class VectorStoreTests(TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        v = rng.normal(size=(60, DIM))
        self.vectors = (v / np.linalg.norm(v, axis=1, keepdims=True)).astype("float32")
        self.texts = [f"chunk {i}" for i in range(60)]

    def test_serialization_roundtrip(self):
        vs = VectorIndex.build(self.vectors, self.texts)
        again = VectorIndex.from_bytes(vs.to_bytes(), self.texts)
        q = self.vectors[3]
        self.assertEqual(vs.mmr_search(q, 3, 15, .5), again.mmr_search(q, 3, 15, .5))

    def test_exact_match_ranks_first(self):
        vs = VectorIndex.build(self.vectors, self.texts)
        row, d2 = vs.mmr_search(self.vectors[42], 3, 15, .5)[0]
        self.assertEqual(row, 42)
        self.assertAlmostEqual(d2, 0.0, places=5)

    def test_mmr_matches_langchain_helper_for_any_lambda(self):
        try:
            from langchain_community.vectorstores.utils import maximal_marginal_relevance as lc_mmr
        except ImportError:
            self.skipTest("langchain-community not installed")
        q = self.vectors[5] + 0.5 * self.vectors[9]
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.assertEqual(maximal_marginal_relevance(q, list(self.vectors[:15]), lambda_mult=lam, k=4),
                             lc_mmr(q, list(self.vectors[:15]), lambda_mult=lam, k=4), lam)

    def test_matches_langchain_faiss_exactly(self):
        """Same documents, same order and same scores as langchain-community's FAISS MMR."""
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.embeddings import Embeddings
        except ImportError:
            self.skipTest("langchain-community not installed")

        class Fixed(Embeddings):
            def embed_documents(self, texts):
                return []

            def embed_query(self, text):
                return []

        lc = FAISS.from_embeddings(list(zip(self.texts, self.vectors.tolist())), Fixed())
        ours = VectorIndex.build(self.vectors, self.texts)
        for qi in (0, 11, 37, 59):
            q = (self.vectors[qi] + 0.3 * self.vectors[(qi + 5) % 60]).tolist()
            theirs = lc.max_marginal_relevance_search_with_score_by_vector(q, k=3, fetch_k=15, lambda_mult=0.5)
            mine = ours.mmr_search(q, 3, 15, 0.5)
            self.assertEqual([d.page_content for d, _ in theirs], [self.texts[r] for r, _ in mine])
            for (_, s1), (_, s2) in zip(theirs, mine):
                self.assertAlmostEqual(float(s1), s2, places=5)

    def test_mmr_helper_edge_cases(self):
        self.assertEqual(maximal_marginal_relevance(np.ones(4), [], k=3), [])
        self.assertEqual(len(maximal_marginal_relevance(np.ones(4), [np.ones(4)], k=3)), 1)


# ------------------------------------------------------------------ API flows

class ApiTests(TestCase):
    def setUp(self):
        self.c = login_client()
        core._indexes.clear()
        patches = [
            mock.patch.object(core, "_embed_batch", side_effect=fake_embed_batch),
            mock.patch.object(core, "_gemini_client", return_value=None),
            mock.patch.object(core, "fetch_video_title", return_value="Chains in LangChain"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def post(self, path, body, **extra):
        return self.c.post(path, json.dumps(body), content_type="application/json", **extra)

    def index_pasted(self, vid="5hjrPILA3-8"):
        r = self.post("/api/sources/manual/", {"url": vid, "text": TRANSCRIPT, "lang_code": "en"},
                      HTTP_ACCEPT="text/event-stream")
        self.assertEqual(r.status_code, 200)
        return sse_events(r)

    def test_health_and_config(self):
        self.assertEqual(self.c.get("/api/health/").json()["database"], True)
        cfg = self.c.get("/api/config/").json()
        self.assertEqual(cfg["retrieval"], {"chunk_size": 1000, "chunk_overlap": 200, "k": 3, "fetch_k": 15,
                                            "lambda_mult": 0.5, "temperature": 0.5})
        self.assertEqual(len(cfg["langs"]), 20)

    def test_settings_validation(self):
        self.assertEqual(self.c.patch("/api/settings/", json.dumps({"max_tokens": 2048}),
                                      content_type="application/json").json()["max_tokens"], 2048)
        bad = self.c.patch("/api/settings/", json.dumps({"max_tokens": 7}), content_type="application/json")
        self.assertEqual(bad.status_code, 400)

    def test_index_pasted_transcript_then_reuse(self):
        events = self.index_pasted()
        names = [n for n, _ in events]
        self.assertEqual(names[0], "meta")
        self.assertEqual(names[-1], "done")
        src = VideoSource.objects.get()
        self.assertGreater(src.n_chunks, 1)
        self.assertEqual(src.chunks.count(), src.n_chunks)
        self.assertTrue(src.index_blob)                         # index lives in Postgres
        self.assertTrue(TranscriptCache.objects.filter(video_id="5hjrPILA3-8").exists())
        again = self.index_pasted()
        self.assertTrue(again[-1][1]["reused"])

    def test_ask_streams_grounded_answer(self):
        self.index_pasted()
        src = VideoSource.objects.get()
        conv = self.post("/api/conversations/", {"source_id": src.id}).json()
        core._indexes.clear()                                   # force rebuild from Postgres (cold serverless start)

        def fake_answer(context, question, max_tokens, stats=None, rules=""):
            self.assertIn("sequential chain", context)          # retrieved transcript reached the prompt
            stats.update({"generation_tokens": 4, "finish_reason": "stop"})
            yield from ["<think>check</think>", "Sequential ", "chains ", "run in order."]

        with mock.patch.object(core, "generate_answer", side_effect=fake_answer), \
                mock.patch.object(core, "_grok_client", return_value=None):
            r = self.post(f"/api/conversations/{conv['id']}/ask/", {"question": "What is a sequential chain?"},
                          HTTP_ACCEPT="text/event-stream")
            events = sse_events(r)
        names = [n for n, _ in events]
        self.assertEqual(names[0], "start")
        self.assertIn("token", names)
        done = dict(events)["done"]["message"]
        self.assertEqual(done["content"], "Sequential chains run in order.")
        self.assertEqual(done["reasoning"], "check")
        self.assertEqual(len(done["trace"]["sources"]), 3)
        self.assertTrue(all(0 <= s["cos"] <= 1 for s in done["trace"]["sources"]))
        self.assertEqual(Message.objects.filter(role="assistant").count(), 1)
        self.assertEqual(QueryMetric.objects.count(), 1)
        self.assertEqual(Conversation.objects.get(pk=conv["id"]).title, "What is a sequential chain?")

    def test_ask_reports_provider_failure_as_message(self):
        self.index_pasted()
        conv = self.post("/api/conversations/", {"source_id": VideoSource.objects.get().id}).json()
        with mock.patch.object(core, "_grok_client", side_effect=core.ConfigError("No chat-model key is set.")):
            events = sse_events(self.post(f"/api/conversations/{conv['id']}/ask/", {"question": "hi there"}))
        err = dict(events)["error"]
        self.assertIn("No chat-model key", err["message"])
        self.assertTrue(Message.objects.filter(role="assistant").exclude(error="").exists())

    def test_youtube_block_is_reported_clearly(self):
        from youtube_transcript_api._errors import RequestBlocked

        class Blocked:
            def list(self, v):
                raise RequestBlocked(v)

            def fetch(self, v, languages):
                raise RequestBlocked(v)

        with mock.patch.object(core, "_yt_api", return_value=Blocked()), \
                mock.patch.object(core, "yt_dlp_available", return_value=False):
            peek = self.post("/api/video/inspect/", {"url": "zzzzzzzzzzz"}).json()
            self.assertTrue(peek["blocked"])
            r = self.post("/api/sources/index/", {"url": "zzzzzzzzzzz"})
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.json()["error"]["code"], "youtube_blocked")

    def test_validation_errors(self):
        self.assertEqual(self.post("/api/video/inspect/", {"url": "https://vimeo.com/1"}).status_code, 400)
        short = self.post("/api/sources/manual/", {"url": "5hjrPILA3-8", "text": "too short", "lang_code": "en"})
        self.assertEqual(short.status_code, 400)
        conv = self.post("/api/conversations/", {}).json()
        self.assertEqual(self.post(f"/api/conversations/{conv['id']}/ask/", {"question": "x"}).status_code, 400)

    def test_event_stream_accept_header_is_not_rejected(self):
        """Regression: browsers send Accept: text/event-stream; DRF used to answer 406."""
        r = self.post("/api/sources/manual/", {"url": "5hjrPILA3-8", "text": TRANSCRIPT, "lang_code": "en"},
                      HTTP_ACCEPT="text/event-stream")
        self.assertNotEqual(r.status_code, 406)


class ServingTests(TestCase):
    """The production service serves the React build and the API from one origin."""

    def test_health_reports_version(self):
        body = Client().get("/api/health/").json()
        self.assertIn("version", body)
        self.assertTrue(body["database"])

    def test_spa_fallback_serves_index_html(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            Path(d, "index.html").write_text("<!doctype html><div id=root></div>")
            with override_settings(FRONTEND_DIST=Path(d)):
                for path in ("/", "/sources", "/some/deep/route"):
                    r = Client().get(path)
                    self.assertEqual(r.status_code, 200, path)
                    self.assertIn(b'id=root', r.content)
                    self.assertEqual(r["Cache-Control"], "no-cache")
        c = login_client()
        self.assertEqual(c.get("/api/config/").status_code, 200)             # API is not swallowed by the SPA route
        self.assertEqual(c.get("/api/does-not-exist/").status_code, 404)


# ------------------------------------------------------------------ Gemini audio-transcription fallback

class GeminiModelPickTests(TestCase):
    def test_newest_stable_flash_first_and_lite_excluded(self):
        class M:
            def __init__(self, name):
                self.name, self.supported_actions = f"models/{name}", ["generateContent"]

        class Client:
            class models:
                @staticmethod
                def list():
                    return [M(n) for n in ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.7-flash",
                                           "gemini-3.8-flash-tts", "gemini-flash-latest", "gemini-3.1-flash-lite"]]

        core._state.pop("video_models", None)
        with mock.patch.object(core, "GEMINI_VIDEO_MODEL", ""):
            order = core._video_model_candidates(Client())
        core._state.pop("video_models", None)
        self.assertEqual(order[0], "gemini-3.1-flash-lite")                    # lite first (fast, rarely overloaded)
        self.assertEqual(order[1:3], ["gemini-3.8-flash", "gemini-3.7-flash"])
        self.assertNotIn("gemini-3.8-flash-tts", order)

    def test_timestamp_noise_is_stripped(self):
        raw = ("<precision> [ 0m3s564ms - 0m3s944ms ] :  Here, <precision> [ 0m3s944ms - 0m5s634ms ] :  we tackle "
               "backpropagation. [00:12] Next part (1:02:33) ends.\n0:45 Last line at 3:30 pm")
        self.assertEqual(core.clean_transcript(raw),
                         "Here, we tackle backpropagation. Next part ends. Last line at 3:30 pm")

    def test_retired_model_detection(self):
        self.assertTrue(core._model_unusable("404 NOT_FOUND. This model is no longer available to new users"))
        self.assertFalse(core._model_unusable("429 RESOURCE_EXHAUSTED"))


class GeminiFallbackTests(TestCase):
    def setUp(self):
        self.account = make_account("gem@example.com")
        core._indexes.clear()
        core._state.pop("yt_blocked_until", None)
        self.addCleanup(core._state.pop, "yt_blocked_until", None)
        for p in [mock.patch.object(core, "_embed_batch", side_effect=fake_embed_batch),
                  mock.patch.object(core, "_gemini_client", return_value=object()),
                  mock.patch.object(core, "fetch_video_title", return_value="T"),
                  mock.patch.object(core, "GEMINI_API_KEY", "test-key"),
                  mock.patch.object(core, "TRANSCRIBE_CHUNK_MIN", 10),
                  mock.patch.object(core, "TRANSCRIBE_MAX_MIN", 30),
                  mock.patch.object(core, "_video_model_candidates", return_value=["m1"])]:
            p.start()
            self.addCleanup(p.stop)

    def test_windows_are_joined_in_order_language_parsed_and_end_detected(self):
        calls = []

        def fake_chunk(client, model, vid, s, e, lang):
            calls.append((s, e, lang))
            if s >= 1200:
                raise RuntimeError("400 INVALID_ARGUMENT. Request contains an invalid argument.")   # past the end
            return {0: "LANG: hi\nनमस्ते दोस्तों आज हम चेन के बारे में बात करेंगे", 600: "दूसरा हिस्सा यहाँ है"}[s]

        with mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk):
            events = list(core.gemini_transcribe("zzzzzzzzzzz"))
        done = events[-1]
        self.assertEqual(done[0], "done")
        _, text, lang, minutes, partial, gap = done
        self.assertEqual(lang, "hi")
        self.assertLess(text.index("नमस्ते"), text.index("दूसरा"))            # window order preserved
        self.assertNotIn("LANG", text)
        self.assertEqual((minutes, partial, gap), (20, False, 0))
        self.assertEqual(sorted(c[:2] for c in calls)[:2], [(0, 600), (600, 1200)])

    def test_marks_partial_when_cap_reached(self):
        with mock.patch.object(core, "_transcribe_chunk",
                               side_effect=lambda *a: "LANG: en\ntext" if a[3] == 0 else "more"):
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
        self.assertTrue(done[4])
        self.assertEqual(done[3], 30)

    def test_blocked_empty_answers_are_retried_not_treated_as_the_end(self):
        """Regression: Gemini's recitation filter returns '' at random; that used to end the transcript early."""
        seen = {}

        def fake_chunk(client, model, vid, s, e, lang):
            seen[s] = seen.get(s, 0) + 1
            if s == 600 and seen[s] == 1:
                return ""                                       # blocked once, fine on the retry
            if s >= 1200:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return "LANG: en\nfirst part" if s == 0 else "second part"

        with mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk):
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
        self.assertEqual(done[1], "first part second part")
        self.assertEqual(seen[600], 2)

    def test_stretch_blocked_at_every_size_leaves_a_reported_gap_but_continues(self):
        def fake_chunk(client, model, vid, s, e, lang):
            if 600 <= s < 1200:
                return ""                                       # blocked however it is sliced
            if s >= 1800:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return "LANG: en\nfirst" if s == 0 else "third"

        with mock.patch.object(core, "_video_model_candidates", return_value=["m1"]), \
                mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk), \
                mock.patch.object(core, "TRANSCRIBE_MAX_MIN", 40):
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
        _, text, lang, minutes, partial, gap = done
        self.assertIn("first", text)
        self.assertIn("third", text)                             # the stretch after the hole is still captured
        self.assertEqual(gap, 10)                                # and the hole is reported, not hidden

    def test_blocked_window_is_redone_as_short_pieces(self):
        """Regression: a window blocked on every retry used to leave a 6-minute hole in the transcript."""
        def fake_chunk(client, model, vid, s, e, lang):
            if s >= 1200:
                raise RuntimeError("400 INVALID_ARGUMENT")
            if e - s > 90:
                return ""                                       # whole window always blocked
            return ("LANG: en\n" if s == 0 else "") + f"piece-{s}"

        with mock.patch.object(core, "_video_model_candidates", return_value=["m1"]), \
                mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk):
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
        _, text, lang, minutes, partial, gap = done
        self.assertEqual(gap, 0)
        self.assertEqual(lang, "en")
        for start in list(range(0, 600, 90)) + list(range(600, 1200, 90)):       # every re-sliced 90 s piece is present
            self.assertIn(f"piece-{start}", text)

    def test_language_falls_back_to_script_when_lang_line_missing(self):
        def fake_chunk(client, model, vid, s, e, lang):
            if s >= 600:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return "नमस्ते दोस्तों आज हम चेन के बारे में बात करेंगे"       # model forgot the LANG line

        with mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk):
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
        self.assertEqual(done[2], "hi")

    def test_all_windows_start_together_not_one_after_another(self):
        """Regression (measured on Render): the first window ran alone and took 5+ minutes."""
        import threading
        started, gate = [], threading.Event()

        def fake_chunk(client, model, vid, s, e, lang):
            started.append(s)
            if len(started) >= 3:
                gate.set()
            gate.wait(5)                                        # a window only finishes once all 3 have started
            if s >= 1200:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return "LANG: en\ntext-%d" % s

        with mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk):
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
        self.assertTrue(gate.is_set(), "windows were not run in parallel")
        self.assertEqual(done[2], "en")

    def test_slow_model_is_raced_by_the_next_one(self):
        """A model that stalls (Gemini 504s after ~80 s) must not delay the window: hedge to the next model."""
        import threading
        release, order = threading.Event(), []

        def fake_chunk(client, model, vid, s, e, lang):
            order.append(model)
            if model == "slow":
                release.wait(10)                                # stalls
                raise RuntimeError("504 DEADLINE_EXCEEDED")
            if s >= 600:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return "LANG: en\nfast answer"

        with mock.patch.object(core, "HEDGE_S", 0.2), \
                mock.patch.object(core, "_video_model_candidates", return_value=["slow", "fast"]), \
                mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk):
            import time as _t
            t0 = _t.time()
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
            took = _t.time() - t0
            release.set()
        self.assertEqual(done[1], "fast answer")
        self.assertLess(took, 5)
        self.assertIn("fast", order)                            # the slow model was raced, not waited for

    def test_known_youtube_block_skips_the_slow_routes(self):
        calls = []

        class Blocked:
            def list(self, v):
                calls.append("list")
                from youtube_transcript_api._errors import IpBlocked
                raise IpBlocked(v)

        with mock.patch.object(core, "_yt_api", return_value=Blocked()), \
                mock.patch.object(core, "yt_dlp_available", return_value=True), \
                mock.patch.object(core, "_ydl_tracks", side_effect=RuntimeError("blocked")) as ydl:
            for vid in ("aaaaaaaaaaa", "bbbbbbbbbbb"):
                with self.assertRaises(core.YouTubeBlocked):
                    core.list_transcript_tracks(vid)
        self.assertEqual(calls, ["list"])                       # 2nd video: no request to YouTube at all
        self.assertEqual(ydl.call_count, 0)                     # yt-dlp is skipped too: same blocked IP, ~40 s saved

    def test_model_that_just_hit_quota_is_put_on_cooldown(self):
        core._state.pop("cool", None)
        self.addCleanup(core._state.pop, "cool", None)

        def fake_chunk(client, model, vid, s, e, lang):
            if model == "m1":
                raise RuntimeError("429 RESOURCE_EXHAUSTED. You exceeded your current quota")
            if s >= 600:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return "LANG: en\nfrom m2"

        with mock.patch.object(core, "_video_model_candidates", return_value=["m1", "m2"]), \
                mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk):
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
        self.assertEqual(done[1], "from m2")
        self.assertGreater(core._state["cool"]["m1"], time.time() + 60)          # quota errors cool down ~90 s
        self.assertEqual(core._pick_model(["m1", "m2"], 0), "m2")                 # and m1 is skipped meanwhile
        self.assertEqual(core._pick_model(["m1", "m2"], 1), "m2")

    def test_overloaded_model_rotates_to_the_next_one(self):
        tried = []

        def fake_chunk(client, model, vid, s, e, lang):
            tried.append((model, s))
            if model == "m1":
                raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
            if s >= 600:
                raise RuntimeError("400 INVALID_ARGUMENT")
            return "LANG: en\nhello world text"

        with mock.patch.object(core, "_video_model_candidates", return_value=["m1", "m2"]), \
                mock.patch.object(core, "_transcribe_chunk", side_effect=fake_chunk):
            done = list(core.gemini_transcribe("zzzzzzzzzzz"))[-1]
        self.assertEqual(done[1], "hello world text")
        self.assertIn(("m1", 0), tried)
        self.assertIn(("m2", 0), tried)                         # window 0 moved on to the next model after the 503

    def test_all_models_failing_raises_clear_error(self):
        with mock.patch.object(core, "_video_model_candidates", return_value=["m1", "m2"]), \
                mock.patch.object(core, "_transcribe_chunk", side_effect=RuntimeError("503 UNAVAILABLE high demand")):
            with self.assertRaises(core.GeminiTranscribeError):
                list(core.gemini_transcribe("zzzzzzzzzzz"))

    def test_no_speech_at_all_is_an_error(self):
        with mock.patch.object(core, "_transcribe_chunk", return_value=core.NO_SPEECH):
            with self.assertRaises(core.GeminiTranscribeError):
                list(core.gemini_transcribe("zzzzzzzzzzz"))

    def test_blocked_youtube_falls_back_to_gemini_and_indexes(self):
        from youtube_transcript_api._errors import RequestBlocked

        class Blocked:
            def list(self, v):
                raise RequestBlocked(v)

            def fetch(self, v, languages):
                raise RequestBlocked(v)

        text = "LANG: en\n" + ("A sequential chain runs steps one after another. " * 30)
        c = login_client(self.account)
        with mock.patch.object(core, "_yt_api", return_value=Blocked()), \
                mock.patch.object(core, "yt_dlp_available", return_value=False), \
                mock.patch.object(core, "_transcribe_chunk", side_effect=lambda *a: text if a[3] == 0 else (_ for _ in ()).throw(RuntimeError("400 INVALID_ARGUMENT"))):
            peek = c.post("/api/video/inspect/", json.dumps({"url": "zzzzzzzzzzz"}), content_type="application/json").json()
            self.assertTrue(peek["blocked"])
            self.assertEqual(peek["fallback"], "gemini")
            r = c.post("/api/sources/index/", json.dumps({"url": "zzzzzzzzzzz"}), content_type="application/json",
                       HTTP_ACCEPT="text/event-stream")
            self.assertEqual(r.status_code, 200)
            events = sse_events(r)
        names = [n for n, _ in events]
        self.assertEqual(names.count("meta"), 1)
        self.assertEqual(names[-1], "done")
        src = VideoSource.objects.get()
        self.assertEqual(src.lang_code, "en")
        self.assertIn("Gemini", src.details["fetch"])
        self.assertTrue(TranscriptCache.objects.filter(video_id="zzzzzzzzzzz", origin="gemini").exists())

    def test_without_gemini_key_blocked_stays_a_503(self):
        from youtube_transcript_api._errors import RequestBlocked

        class Blocked:
            def list(self, v):
                raise RequestBlocked(v)

        with mock.patch.object(core, "GEMINI_API_KEY", ""), mock.patch.object(core, "_yt_api", return_value=Blocked()), \
                mock.patch.object(core, "yt_dlp_available", return_value=False):
            r = login_client(self.account).post("/api/sources/index/", json.dumps({"url": "yyyyyyyyyyy"}),
                                                content_type="application/json")
        self.assertEqual(r.status_code, 503)



# ------------------------------------------------------------------ email one-time-code sign-in

import re as _re
from datetime import timedelta as _td

from django.core import mail
from django.utils import timezone as _tz

from . import mailer
from .auth import OTP_MAX_ATTEMPTS, RESEND_COOLDOWN_S, SESSION_COOKIE


def _code_from_outbox(index=-1):
    m = _re.search(r"\b(\d{6})\b", mail.outbox[index].subject)
    assert m, mail.outbox[index].subject
    return m.group(1)


class AuthFlowTests(TestCase):
    def setUp(self):
        self.c = Client()
        mail.outbox = []
        from django.core.cache import cache
        cache.clear()

    def post(self, path, body, client=None, **extra):
        return (client or self.c).post(path, json.dumps(body), content_type="application/json", **extra)

    def start(self, email="new@example.com", password=PASSWORD, client=None):
        return self.post("/api/auth/start/", {"email": email, "password": password}, client)

    def test_api_requires_login_but_health_is_open(self):
        self.assertEqual(self.c.get("/api/health/").status_code, 200)
        r = self.c.get("/api/config/")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"]["code"], "auth_required")
        self.assertEqual(self.c.get("/api/auth/me/").status_code, 401)

    def test_signup_with_emailed_code_creates_account_and_session(self):
        r = self.start("New.User@Example.com")
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(body["purpose"], "signup")
        self.assertEqual(body["email"], "new.user@example.com")                 # normalised
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ["new.user@example.com"])
        self.assertIn("Janani", mail.outbox[0].from_email)
        self.assertFalse(Account.objects.filter(email="new.user@example.com").exists())   # not yet: email unverified
        code = _code_from_outbox()
        ok = self.post("/api/auth/verify/", {"challenge_id": body["challenge_id"], "code": code})
        self.assertEqual(ok.status_code, 200, ok.content)
        self.assertEqual(ok.json()["account"]["email"], "new.user@example.com")
        cookie = ok.cookies[SESSION_COOKIE]
        self.assertTrue(cookie["httponly"])
        self.assertEqual(cookie["samesite"], "Lax")
        acct = Account.objects.get(email="new.user@example.com")
        self.assertNotIn(PASSWORD, acct.password_hash)                          # hashed, never plain
        self.assertEqual(self.c.get("/api/auth/me/").json()["account"]["email"], "new.user@example.com")
        self.assertEqual(self.c.get("/api/config/").status_code, 200)

    def test_login_with_password_then_code(self):
        make_account("me@example.com")
        r = self.start("me@example.com")
        self.assertEqual(r.json()["purpose"], "login")
        ok = self.post("/api/auth/verify/", {"challenge_id": r.json()["challenge_id"], "code": _code_from_outbox()})
        self.assertEqual(ok.status_code, 200)
        self.assertIsNotNone(Account.objects.get(email="me@example.com").last_login)

    def test_wrong_password_sends_no_email(self):
        make_account("me@example.com")
        r = self.start("me@example.com", "wrong-password-1")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"]["code"], "invalid_credentials")
        self.assertEqual(len(mail.outbox), 0)

    def test_repeated_wrong_passwords_are_throttled(self):
        make_account("me@example.com")
        codes = [self.start("me@example.com", f"wrong-password-{i}").status_code for i in range(10)]
        self.assertEqual(codes[:8], [401] * 8)
        self.assertEqual(codes[8], 429)
        self.assertEqual(self.start("me@example.com", PASSWORD).status_code, 429)   # even the right one waits

    def test_weak_and_invalid_input_rejected(self):
        self.assertEqual(self.start("not-an-email").status_code, 400)
        self.assertEqual(self.start("a@example.com", "short").status_code, 400)
        self.assertEqual(self.start("a@example.com", "12345678").status_code, 400)
        self.assertEqual(self.post("/api/auth/start/", {"email": "a@example.com"}).status_code, 400)
        self.assertEqual(len(mail.outbox), 0)

    def test_wrong_code_locks_after_max_attempts(self):
        cid = self.start().json()["challenge_id"]
        good = _code_from_outbox()
        bad = "000000" if good != "000000" else "111111"
        left = []
        for _ in range(OTP_MAX_ATTEMPTS):
            r = self.post("/api/auth/verify/", {"challenge_id": cid, "code": bad})
            self.assertEqual(r.status_code, 400)
            left.append(r.json()["error"]["attempts_left"])
        self.assertEqual(left, [4, 3, 2, 1, 0])
        r = self.post("/api/auth/verify/", {"challenge_id": cid, "code": good})       # right code, too late
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()["error"]["code"], "challenge_invalid")
        self.assertFalse(Account.objects.filter(email="new@example.com").exists())

    def test_code_is_single_use_and_expires(self):
        cid = self.start().json()["challenge_id"]
        code = _code_from_outbox()
        self.assertEqual(self.post("/api/auth/verify/", {"challenge_id": cid, "code": code}).status_code, 200)
        again = self.post("/api/auth/verify/", {"challenge_id": cid, "code": code}, Client())
        self.assertEqual(again.status_code, 400)
        self.assertEqual(again.json()["error"]["code"], "challenge_invalid")
        # expiry
        mail.outbox = []
        c2 = Client()
        cid2 = self.start("late@example.com", client=c2).json()["challenge_id"]
        LoginChallenge.objects.filter(pk=cid2).update(expires_at=_tz.now() - _td(seconds=1))
        r = self.post("/api/auth/verify/", {"challenge_id": cid2, "code": _code_from_outbox()}, c2)
        self.assertEqual(r.json()["error"]["code"], "code_expired")

    def test_code_is_stored_hashed_only(self):
        cid = self.start().json()["challenge_id"]
        code = _code_from_outbox()
        ch = LoginChallenge.objects.get(pk=cid)
        self.assertNotIn(code, ch.code_hash)
        self.assertEqual(len(ch.code_hash), 64)
        self.assertNotIn(PASSWORD, ch.password_hash)

    def test_resend_cooldown_then_new_code_replaces_old(self):
        cid = self.start().json()["challenge_id"]
        old = _code_from_outbox()
        r = self.post("/api/auth/resend/", {"challenge_id": cid})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["error"]["code"], "cooldown")
        LoginChallenge.objects.filter(pk=cid).update(last_sent=_tz.now() - _td(seconds=RESEND_COOLDOWN_S + 1))
        self.assertEqual(self.post("/api/auth/resend/", {"challenge_id": cid}).status_code, 200)
        self.assertEqual(len(mail.outbox), 2)
        new = _code_from_outbox()
        if new != old:
            self.assertEqual(self.post("/api/auth/verify/", {"challenge_id": cid, "code": old}).status_code, 400)
        self.assertEqual(self.post("/api/auth/verify/", {"challenge_id": cid, "code": new}).status_code, 200)

    def test_requesting_a_new_start_invalidates_the_older_code(self):
        c = Client()
        first = self.start(client=c).json()["challenge_id"]
        LoginChallenge.objects.filter(pk=first).update(last_sent=_tz.now() - _td(seconds=RESEND_COOLDOWN_S + 1))
        second = self.start(client=c).json()["challenge_id"]
        self.assertNotEqual(first, second)
        r = self.post("/api/auth/verify/", {"challenge_id": first, "code": _code_from_outbox(0)}, c)
        self.assertEqual(r.json()["error"]["code"], "challenge_invalid")

    def test_logout_ends_the_session(self):
        c = login_client()
        self.assertEqual(c.get("/api/config/").status_code, 200)
        self.assertEqual(c.post("/api/auth/logout/", "{}", content_type="application/json").status_code, 204)
        self.assertEqual(c.get("/api/config/").status_code, 401)

    def test_expired_session_is_rejected(self):
        c = login_client()
        AuthSession.objects.update(expires_at=_tz.now() - _td(seconds=1))
        self.assertEqual(c.get("/api/config/").status_code, 401)

    def test_cross_site_requests_are_blocked(self):
        c = login_client()
        evil = c.post("/api/conversations/", "{}", content_type="application/json", HTTP_ORIGIN="https://evil.example")
        self.assertEqual(evil.status_code, 403)
        self.assertEqual(evil.json()["error"]["code"], "bad_origin")
        same = c.post("/api/conversations/", "{}", content_type="application/json", HTTP_ORIGIN="http://testserver")
        self.assertIn(same.status_code, (200, 201))
        # a dev-server origin is only trusted while DEBUG is on
        dev = dict(content_type="application/json", HTTP_ORIGIN="http://127.0.0.1:5180")
        with override_settings(DEBUG=False):
            self.assertEqual(c.post("/api/conversations/", "{}", **dev).status_code, 403)
        with override_settings(DEBUG=True):
            self.assertIn(c.post("/api/conversations/", "{}", **dev).status_code, (200, 201))
        fwd = c.post("/api/conversations/", "{}", content_type="application/json",
                     HTTP_ORIGIN="https://janani.pvsairamsaketh.in", HTTP_X_FORWARDED_HOST="janani.pvsairamsaketh.in")
        self.assertIn(fwd.status_code, (200, 201))

    def test_invite_list_restricts_signups(self):
        with override_settings(ALLOWED_EMAILS={"ok@example.com"}, ALLOWED_EMAIL_DOMAINS={"corp.test"}):
            self.assertEqual(self.start("stranger@example.com").status_code, 403)
            self.assertEqual(self.start("ok@example.com").status_code, 200)
            self.assertEqual(self.start("boss@corp.test", client=Client()).status_code, 200)

    def test_only_the_owner_adopts_pre_login_data(self):
        orphan = Conversation.objects.create(title="from before sign-in")
        src = VideoSource.objects.create(key="v:en", video_id="aaaaaaaaaaa", lang_code="en", lang_name="English",
                                         lang_kind="manual", index_file="x")
        # a stranger who signs up FIRST gets nothing
        cid0 = self.start("stranger@example.com", client=Client()).json()["challenge_id"]
        c0 = Client()
        self.post("/api/auth/verify/", {"challenge_id": cid0, "code": _code_from_outbox()}, c0)
        self.assertEqual(c0.get("/api/conversations/").json()["results"], [])
        orphan.refresh_from_db()
        self.assertIsNone(orphan.account_id)
        mail.outbox = []
        with override_settings(OWNER_EMAIL="first@example.com"):
            cid = self.start("first@example.com").json()["challenge_id"]
            self.post("/api/auth/verify/", {"challenge_id": cid, "code": _code_from_outbox()})
        acct = Account.objects.get(email="first@example.com")
        orphan.refresh_from_db()
        self.assertEqual(orphan.account_id, acct.id)
        self.assertTrue(UserSource.objects.filter(account=acct, source=src).exists())
        # a second person does NOT get the first person's data
        mail.outbox = []
        c2 = Client()
        cid2 = self.start("second@example.com", client=c2).json()["challenge_id"]
        self.post("/api/auth/verify/", {"challenge_id": cid2, "code": _code_from_outbox()}, c2)
        self.assertEqual(c2.get("/api/conversations/").json()["results"], [])
        self.assertEqual(c2.get("/api/sources/").json()["results"], [])

    def test_email_provider_failure_returns_503_and_leaves_no_challenge(self):
        with mock.patch.object(mailer, "send_otp", side_effect=mailer.EmailSendError("boom", "hint")):
            r = self.start()
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "email_failed")
        self.assertNotIn("boom", r.content.decode())                           # internal detail never leaks
        self.assertEqual(LoginChallenge.objects.count(), 0)

    def test_email_not_configured_returns_clear_503(self):
        with mock.patch.object(mailer, "send_otp", side_effect=mailer.EmailNotConfigured("no key")):
            r = self.start()
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["code"], "email_not_configured")

    def test_health_reports_mail_setup_without_secrets(self):
        body = self.c.get("/api/health/").json()
        self.assertIn("provider", body["email"])
        self.assertNotIn("password", json.dumps(body).lower())


class DataIsolationTests(TestCase):
    """Two people must never see or touch each other's chats or videos."""

    def setUp(self):
        self.a = login_client(make_account("a@example.com"))
        self.b = login_client(make_account("b@example.com"))
        for p in [mock.patch.object(core, "_embed_batch", side_effect=fake_embed_batch),
                  mock.patch.object(core, "_gemini_client", return_value=None),
                  mock.patch.object(core, "fetch_video_title", return_value="T")]:
            p.start()
            self.addCleanup(p.stop)
        core._indexes.clear()

    def post(self, c, path, body, **kw):
        return c.post(path, json.dumps(body), content_type="application/json", **kw)

    def index(self, c, vid="5hjrPILA3-8"):
        return sse_events(self.post(c, "/api/sources/manual/", {"url": vid, "text": TRANSCRIPT, "lang_code": "en"}))

    def test_conversations_are_private(self):
        self.index(self.a)
        src = self.a.get("/api/sources/").json()["results"][0]
        conv = self.post(self.a, "/api/conversations/", {"source_id": src["id"]}).json()
        self.assertEqual(self.b.get(f"/api/conversations/{conv['id']}/").status_code, 404)
        self.assertEqual(self.post(self.b, f"/api/conversations/{conv['id']}/clear/", {}).status_code, 404)
        self.assertEqual(self.b.delete(f"/api/conversations/{conv['id']}/").status_code, 404)
        self.assertEqual(self.post(self.b, f"/api/conversations/{conv['id']}/ask/", {"question": "hello there"}).status_code, 404)
        self.assertEqual(self.b.get("/api/conversations/").json()["results"], [])
        self.assertEqual(self.a.get(f"/api/conversations/{conv['id']}/").status_code, 200)

    def test_sources_are_listed_per_person_but_the_index_is_shared(self):
        self.index(self.a)
        self.assertEqual(self.b.get("/api/sources/").json()["results"], [])
        self.assertEqual(VideoSource.objects.count(), 1)
        r = self.index(self.b)                                                    # same video: reused, no second index
        self.assertTrue(r[-1][1]["reused"])
        self.assertEqual(VideoSource.objects.count(), 1)
        self.assertEqual(len(self.b.get("/api/sources/").json()["results"]), 1)
        sid = self.a.get("/api/sources/").json()["results"][0]["id"]
        # B cannot start a chat on a source B has not added
        c3 = login_client(make_account("c@example.com"))
        self.assertEqual(self.post(c3, "/api/conversations/", {"source_id": sid}).status_code, 404)

    def test_removing_a_source_keeps_it_for_the_other_person(self):
        self.index(self.a)
        self.index(self.b)
        sid = self.a.get("/api/sources/").json()["results"][0]["id"]
        self.assertEqual(self.a.delete(f"/api/sources/{sid}/").status_code, 204)
        self.assertEqual(self.a.get("/api/sources/").json()["results"], [])
        self.assertEqual(len(self.b.get("/api/sources/").json()["results"]), 1)
        self.assertTrue(VideoSource.objects.filter(pk=sid).exists())              # still used by B
        self.assertEqual(self.b.delete(f"/api/sources/{sid}/").status_code, 204)
        self.assertFalse(VideoSource.objects.filter(pk=sid).exists())             # last person frees the index

    def test_cannot_delete_a_source_you_never_added(self):
        self.index(self.a)
        sid = self.a.get("/api/sources/").json()["results"][0]["id"]
        self.assertEqual(self.b.delete(f"/api/sources/{sid}/").status_code, 404)
        self.assertTrue(VideoSource.objects.filter(pk=sid).exists())

    def test_settings_are_per_person(self):
        self.assertEqual(self.a.patch("/api/settings/", json.dumps({"max_tokens": 2048}),
                                      content_type="application/json").json()["max_tokens"], 2048)
        self.assertEqual(self.b.get("/api/settings/").json()["max_tokens"], 1024)

    def test_delete_all_chats_only_deletes_my_chats(self):
        self.post(self.a, "/api/conversations/", {})
        self.post(self.b, "/api/conversations/", {})
        self.assertEqual(self.a.delete("/api/conversations/").status_code, 204)
        self.assertEqual(Conversation.objects.filter(account=self.b.account).count(), 1)
        self.assertEqual(Conversation.objects.filter(account=self.a.account).count(), 0)

    def test_shared_transcript_cache_cannot_be_wiped_by_a_user(self):
        TranscriptCache.objects.create(video_id="zzzzzzzzzzz", lang_code="en", text="keep me", origin="gemini")
        self.assertEqual(self.post(self.a, "/api/cache/clear/", {}).status_code, 200)
        self.assertTrue(TranscriptCache.objects.filter(video_id="zzzzzzzzzzz").exists())


# ------------------------------------------------------------------ mailer: providers, HTML, real SMTP protocol

class MailerTests(TestCase):
    def setUp(self):
        mail.outbox = []

    def test_message_contains_code_and_expiry_in_text_and_html(self):
        subject, text, html = mailer._message("482913", "login", 10)
        self.assertIn("482913", subject)
        self.assertIn("482913", text)
        self.assertIn("10 minutes", text)
        for digit in "482913":
            self.assertIn(f">{digit}</span>", html)
        self.assertIn("creating", mailer._message("111111", "signup", 10)[1].lower())

    def test_sends_html_and_plain_parts_from_the_configured_sender(self):
        with override_settings(EMAIL_PROVIDER="smtp", DEFAULT_FROM_EMAIL="Janani <pvsaketh1@gmail.com>"):
            mailer.send_otp("someone@example.com", "123456", "login")
        m = mail.outbox[0]
        self.assertEqual(m.from_email, "Janani <pvsaketh1@gmail.com>")
        self.assertEqual(m.to, ["someone@example.com"])
        self.assertIn("123456", m.body)
        self.assertEqual(m.alternatives[0][1], "text/html")

    def test_brevo_https_provider_posts_expected_payload(self):
        seen = {}

        class Resp:
            status = 201

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            seen["url"], seen["headers"], seen["body"] = req.full_url, dict(req.header_items()), json.loads(req.data)
            return Resp()

        with override_settings(EMAIL_PROVIDER="brevo", BREVO_API_KEY="brevo-key", DEFAULT_FROM_EMAIL_ADDRESS="pvsaketh1@gmail.com"), \
                mock.patch("urllib.request.urlopen", fake_urlopen):
            mailer.send_otp("someone@example.com", "654321", "signup")
        self.assertEqual(seen["url"], "https://api.brevo.com/v3/smtp/email")
        self.assertEqual(seen["headers"]["Api-key"], "brevo-key")
        self.assertEqual(seen["body"]["to"], [{"email": "someone@example.com"}])
        self.assertEqual(seen["body"]["sender"]["email"], "pvsaketh1@gmail.com")
        self.assertIn("654321", seen["body"]["textContent"])

    def test_smtp_failures_are_translated_with_actionable_hints(self):
        import smtplib
        with override_settings(EMAIL_PROVIDER="smtp"):
            with mock.patch("django.core.mail.EmailMultiAlternatives.send", side_effect=smtplib.SMTPAuthenticationError(535, b"bad")):
                with self.assertRaises(mailer.EmailSendError) as cm:
                    mailer.send_otp("x@example.com", "123456")
                self.assertIn("App Password", cm.exception.hint)
            with mock.patch("django.core.mail.EmailMultiAlternatives.send", side_effect=ConnectionRefusedError("blocked")):
                with self.assertRaises(mailer.EmailSendError) as cm:
                    mailer.send_otp("x@example.com", "123456")
                self.assertIn("FREE", cm.exception.hint)                          # the Render free-plan SMTP block

    def test_unconfigured_production_refuses_to_pretend(self):
        with override_settings(EMAIL_PROVIDER="console", DEBUG=False, TESTING=False):
            with self.assertRaises(mailer.EmailNotConfigured):
                mailer.send_otp("x@example.com", "123456")
        with override_settings(EMAIL_PROVIDER="smtp", EMAIL_HOST_PASSWORD="", TESTING=False):
            with self.assertRaises(mailer.EmailNotConfigured):
                mailer.send_otp("x@example.com", "123456")

    def test_real_smtp_conversation_against_a_local_server(self):
        """Speaks actual SMTP (EHLO / MAIL FROM / RCPT TO / DATA) to a tiny in-process server."""
        import socket
        import threading
        received = {}
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        def serve():
            conn, _ = srv.accept()
            f = conn.makefile("rwb", buffering=0)
            f.write(b"220 fake ESMTP\r\n")
            data_mode, buf = False, []
            while True:
                line = f.readline()
                if not line:
                    break
                if data_mode:
                    if line == b".\r\n":
                        received["data"] = b"".join(buf).decode("utf-8", "replace")
                        f.write(b"250 queued\r\n")
                        data_mode = False
                    else:
                        buf.append(line)
                    continue
                cmd = line.decode().strip().upper()
                if cmd.startswith("EHLO") or cmd.startswith("HELO"):
                    f.write(b"250 fake\r\n")
                elif cmd.startswith("MAIL FROM"):
                    received["from"] = line.decode().strip()
                    f.write(b"250 ok\r\n")
                elif cmd.startswith("RCPT TO"):
                    received["to"] = line.decode().strip()
                    f.write(b"250 ok\r\n")
                elif cmd == "DATA":
                    data_mode = True
                    f.write(b"354 go\r\n")
                elif cmd == "QUIT":
                    f.write(b"221 bye\r\n")
                    break
                else:
                    f.write(b"250 ok\r\n")
            conn.close()

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        with override_settings(EMAIL_PROVIDER="smtp", EMAIL_BACKEND="django.core.mail.backends.smtp.EmailBackend",
                               EMAIL_HOST="127.0.0.1", EMAIL_PORT=port, EMAIL_USE_TLS=False, EMAIL_USE_SSL=False,
                               EMAIL_HOST_USER="", EMAIL_HOST_PASSWORD="",
                               DEFAULT_FROM_EMAIL="Janani <pvsaketh1@gmail.com>"):
            mailer.send_otp("login@example.com", "907153", "login")
        t.join(5)
        srv.close()
        self.assertIn("pvsaketh1@gmail.com", received["from"])
        self.assertIn("login@example.com", received["to"])
        self.assertIn("907153", received["data"])
        self.assertIn("text/html", received["data"])


# ------------------------------------------------------------------ Gmail API (OAuth over HTTPS) provider

class GmailApiTests(TestCase):
    GMAIL = dict(EMAIL_PROVIDER="gmail_api", GMAIL_CLIENT_ID="cid", GMAIL_CLIENT_SECRET="csecret",
                 GMAIL_REFRESH_TOKEN="rtoken", DEFAULT_FROM_EMAIL="Janani <pvsaketh1@gmail.com>")

    def setUp(self):
        mailer._gmail_token.update(value="", expires=0.0)
        self.addCleanup(mailer._gmail_token.update, value="", expires=0.0)

    class _Resp:
        def __init__(self, body=b"{}", status=200):
            self._b, self.status = body, status

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake(self, calls, fail_send=None):
        def fake_urlopen(req, timeout=None):
            calls.append((req.full_url, dict(req.header_items()), req.data))
            if "oauth2.googleapis.com" in req.full_url:
                return self._Resp(json.dumps({"access_token": "ya29.token", "expires_in": 3600}).encode())
            if fail_send:
                raise fail_send
            return self._Resp(b'{"id":"abc"}')
        return fake_urlopen

    def test_sends_a_real_rfc822_message_through_the_gmail_api(self):
        import base64
        from email import message_from_bytes
        calls = []
        with override_settings(**self.GMAIL), mock.patch("urllib.request.urlopen", self._fake(calls)):
            mailer.send_otp("login@example.com", "314159", "login")
        token_call, send_call = calls
        self.assertIn("grant_type=refresh_token", token_call[2].decode())
        self.assertIn("refresh_token=rtoken", token_call[2].decode())
        self.assertEqual(send_call[0], "https://gmail.googleapis.com/gmail/v1/users/me/messages/send")
        self.assertEqual(send_call[1]["Authorization"], "Bearer ya29.token")
        raw = json.loads(send_call[2])["raw"]
        msg = message_from_bytes(base64.urlsafe_b64decode(raw))
        self.assertEqual(msg["To"], "login@example.com")
        self.assertIn("pvsaketh1@gmail.com", msg["From"])
        self.assertIn("314159", msg["Subject"])
        types = [p.get_content_type() for p in msg.walk()]
        self.assertIn("text/plain", types)
        self.assertIn("text/html", types)

    def test_access_token_is_cached_between_sends(self):
        calls = []
        with override_settings(**self.GMAIL), mock.patch("urllib.request.urlopen", self._fake(calls)):
            mailer.send_otp("a@example.com", "111111")
            mailer.send_otp("b@example.com", "222222")
        self.assertEqual(sum("oauth2.googleapis.com" in c[0] for c in calls), 1)       # one token request
        self.assertEqual(sum("messages/send" in c[0] for c in calls), 2)

    def test_expired_refresh_token_gives_a_publish_the_app_hint(self):
        import io
        import urllib.error

        def boom(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 400, "bad", {}, io.BytesIO(b'{"error":"invalid_grant"}'))
        with override_settings(**self.GMAIL), mock.patch("urllib.request.urlopen", boom):
            with self.assertRaises(mailer.EmailSendError) as cm:
                mailer.send_otp("a@example.com", "111111")
        self.assertIn("7 days", cm.exception.hint)

    def test_api_disabled_or_missing_scope_hint(self):
        import io
        import urllib.error
        err = urllib.error.HTTPError("u", 403, "forbidden", {}, io.BytesIO(b'{"error":{"message":"insufficient"}}'))
        with override_settings(**self.GMAIL), mock.patch("urllib.request.urlopen", self._fake([], fail_send=err)):
            with self.assertRaises(mailer.EmailSendError) as cm:
                mailer.send_otp("a@example.com", "111111")
        self.assertIn("gmail.send", cm.exception.hint)

    def test_missing_credentials_is_not_configured(self):
        with override_settings(EMAIL_PROVIDER="gmail_api", GMAIL_CLIENT_ID="", GMAIL_CLIENT_SECRET="", GMAIL_REFRESH_TOKEN=""):
            with self.assertRaises(mailer.EmailNotConfigured):
                mailer.send_otp("a@example.com", "111111")

    def test_status_reports_provider_without_secrets(self):
        with override_settings(**self.GMAIL):
            st = mailer.status()
        self.assertEqual(st["provider"], "gmail_api")
        self.assertTrue(st["configured"])
        self.assertNotIn("rtoken", json.dumps(st))


class GmailAuthorizeScriptTests(TestCase):
    def _mod(self):
        import importlib.util
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "scripts" / "gmail_authorize.py"
        spec = importlib.util.spec_from_file_location("gmail_authorize", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_env_writer_replaces_or_appends_without_touching_other_lines(self):
        import tempfile
        from pathlib import Path
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, ".env")
            p.write_text('A=1\n# comment\nGMAIL_CLIENT_ID="x"\nGMAIL_REFRESH_TOKEN=old\nB=2\n')
            m.write_env_value("GMAIL_REFRESH_TOKEN", "new-token", p)
            self.assertEqual(p.read_text(), 'A=1\n# comment\nGMAIL_CLIENT_ID="x"\nGMAIL_REFRESH_TOKEN=new-token\nB=2\n')
            m.write_env_value("OTHER", "v", p)
            self.assertTrue(p.read_text().endswith("OTHER=v\n"))
            self.assertEqual(oct(p.stat().st_mode)[-3:], "600")                        # secrets file is owner-only
            self.assertEqual(m.read_env(p)["GMAIL_CLIENT_ID"], "x")
