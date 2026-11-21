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


class Account(models.Model):
    """A person who signed in. Email ownership is proven by a one-time code at sign-up and at every login."""
    email = models.EmailField(max_length=254, unique=True)              # always stored lowercase
    password_hash = models.CharField(max_length=256)
    is_active = models.BooleanField(default=True)
    created = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.email


class LoginChallenge(models.Model):
    """One pending email one-time code (login or sign-up). The code itself is never stored, only an HMAC of it."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(max_length=254, db_index=True)
    purpose = models.CharField(max_length=8)                            # login | signup
    code_hash = models.CharField(max_length=64)
    password_hash = models.CharField(max_length=256, blank=True, default="")   # sign-up only: becomes the account password
    attempts = models.PositiveSmallIntegerField(default=0)              # wrong guesses so far
    sends = models.PositiveSmallIntegerField(default=1)                 # emails sent for this challenge
    last_sent = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    consumed_at = models.DateTimeField(null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)
    ip = models.CharField(max_length=64, blank=True, default="")


class AuthSession(models.Model):
    """Browser session. Only a SHA-256 of the random token is stored; the token lives in an HttpOnly cookie."""
    account = models.ForeignKey(Account, related_name="sessions", on_delete=models.CASCADE)
    token_hash = models.CharField(max_length=64, unique=True)
    created = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    last_used = models.DateTimeField(auto_now_add=True)
    user_agent = models.CharField(max_length=200, blank=True, default="")


class UserSource(models.Model):
    """Which video indexes a person has added. The FAISS index itself is shared (and reused) between people."""
    account = models.ForeignKey(Account, related_name="user_sources", on_delete=models.CASCADE)
    source = models.ForeignKey(VideoSource, related_name="user_links", on_delete=models.CASCADE)
    added = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["account", "source"], name="uniq_account_source")]


class Conversation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    account = models.ForeignKey(Account, null=True, blank=True, related_name="conversations", on_delete=models.CASCADE)
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
    account = models.ForeignKey(Account, null=True, blank=True, on_delete=models.CASCADE)
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

    account = models.OneToOneField(Account, null=True, blank=True, related_name="settings", on_delete=models.CASCADE)

    @classmethod
    def load(cls, account=None):
        """Per-account settings; the old global row (pk=1) is only used when no account is given."""
        if account is not None:
            obj, _ = cls.objects.get_or_create(account=account)
            return obj
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
