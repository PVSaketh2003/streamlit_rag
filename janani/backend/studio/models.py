import uuid

from django.db import models


class VideoSource(models.Model):
    """One FAISS index = one video in one caption language. Chunks live in Postgres,
    the raw FAISS vectors on disk (`index_file`)."""
    key = models.CharField(max_length=64, unique=True)          # "<video_id>:<lang>"
    video_id = models.CharField(max_length=16, db_index=True)
    title = models.CharField(max_length=300, blank=True, default="")
    lang_code = models.CharField(max_length=16)
    lang_name = models.CharField(max_length=64)
    lang_kind = models.CharField(max_length=16)                  # manual | auto-generated
    lang_reason = models.CharField(max_length=200, blank=True, default="")
    tracks = models.JSONField(default=list)
    n_chunks = models.PositiveIntegerField(default=0)
    n_chars = models.PositiveIntegerField(default=0)
    n_words = models.PositiveIntegerField(default=0)
    timings = models.JSONField(default=dict)
    details = models.JSONField(default=dict)
    total_s = models.FloatField(default=0.0)
    index_file = models.CharField(max_length=300)
    index_blob = models.BinaryField(null=True, blank=True)   # serialized FAISS index (serverless-safe)
    embed_model = models.CharField(max_length=80, default="legacy")
    built_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-built_at"]

    def __str__(self):
        return f"{self.title or self.video_id} [{self.lang_code}]"


class Chunk(models.Model):
    source = models.ForeignKey(VideoSource, related_name="chunks", on_delete=models.CASCADE)
    idx = models.PositiveIntegerField()                          # position == FAISS vector id
    text = models.TextField()

    class Meta:
        ordering = ["idx"]
        constraints = [models.UniqueConstraint(fields=["source", "idx"], name="uniq_chunk_per_source")]


class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=120, default="New conversation")
    source = models.ForeignKey(VideoSource, null=True, blank=True, on_delete=models.SET_NULL,
                               related_name="conversations")
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated"]


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, related_name="messages", on_delete=models.CASCADE)
    role = models.CharField(max_length=12)                       # user | assistant
    content = models.TextField(blank=True, default="")
    reasoning = models.TextField(blank=True, default="")
    error = models.TextField(blank=True, default="")
    trace = models.JSONField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]


class QueryMetric(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    video_id = models.CharField(max_length=16)
    retrieval_s = models.FloatField(null=True)
    model_load_s = models.FloatField(null=True)
    ttft_s = models.FloatField(null=True)
    gen_s = models.FloatField(null=True)
    tokens = models.IntegerField(null=True)
    tps = models.FloatField(null=True)
    total_s = models.FloatField(null=True)

    class Meta:
        ordering = ["-created"]


class AppSettings(models.Model):
    """Singleton row (pk=1)."""
    max_tokens = models.PositiveIntegerField(default=1024)
    answer_lang = models.CharField(max_length=8, default="en")
    video_lang = models.CharField(max_length=16, default="auto")
    bridge_boost = models.BooleanField(default=True)
    show_reasoning = models.BooleanField(default=False)

    @classmethod
    def load(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class TranscriptCache(models.Model):
    """Every transcript fetched once is kept, so YouTube is never asked twice for the same captions."""
    video_id = models.CharField(max_length=16)
    lang_code = models.CharField(max_length=16)
    text = models.TextField()
    origin = models.CharField(max_length=16, default="youtube")   # youtube | yt-dlp | manual
    fetched_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["video_id", "lang_code"], name="uniq_transcript")]


class TrackCache(models.Model):
    video_id = models.CharField(max_length=16, unique=True)
    tracks = models.JSONField(default=list)
    fetched_at = models.DateTimeField(auto_now=True)
