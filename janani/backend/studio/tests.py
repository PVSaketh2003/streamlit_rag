"""Backend test suite (run: `python manage.py test studio`). External services (YouTube, Gemini, Groq/xAI)
are mocked, so the suite is deterministic and needs only PostgreSQL."""
import hashlib
import json
from unittest import mock

import numpy as np
from django.test import Client, TestCase, override_settings

from . import rag_core as core
from .models import Conversation, Message, QueryMetric, TranscriptCache, VideoSource
from .vstore import VectorIndex, maximal_marginal_relevance

DIM = 32


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

@override_settings(APP_ACCESS_TOKEN="")
class ApiTests(TestCase):
    def setUp(self):
        self.c = Client()
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


@override_settings(APP_ACCESS_TOKEN="s3cret-code")
class AccessGateTests(TestCase):
    def test_requires_token_except_health(self):
        c = Client()
        self.assertEqual(c.get("/api/health/").status_code, 200)
        r = c.get("/api/config/")
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()["error"]["code"], "access_required")
        self.assertEqual(c.get("/api/config/", HTTP_X_ACCESS_TOKEN="wrong").status_code, 401)
        self.assertEqual(c.get("/api/config/", HTTP_X_ACCESS_TOKEN="s3cret-code").status_code, 200)
