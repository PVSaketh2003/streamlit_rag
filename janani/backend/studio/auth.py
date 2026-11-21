"""Email + password sign-in with an emailed one-time code (OTP).

JWT based flow
  1. POST /api/auth/start/   {email, password}
        known email   -> password checked, a login code is emailed
        unknown email -> a sign-up code is emailed; the account is created only after the code is verified
     The response carries `challenge_token`: a signed JWT (HS256, 10 min, audience "janani:otp") whose claims are
     {sub: email, jti: challenge id, pur: login|signup, otp: HMAC(code)}. It never contains the code, the password or
     its hash; the HMAC is keyed with a server secret so the 6 digits cannot be brute-forced offline.
  2. POST /api/auth/verify/  {challenge_token, code}
     The signature, expiry, issuer and audience of the JWT are verified, then the code is checked against the `otp`
     claim, and a server-side row enforces single use and the 5-guess limit. Success returns a *session JWT*
     (audience "janani:session", 30 days) in an HttpOnly cookie (also accepted as `Authorization: Bearer`).
  3. POST /api/auth/resend/  {challenge_token}  -> a new code and a new challenge_token (the old one stops working)
  4. GET  /api/auth/me/, POST /api/auth/logout/  (logout revokes the session's `jti` server-side)

Security properties: codes are 6 digits from `secrets`, valid 10 minutes, single use, max 5 wrong guesses, resend
cooldown, per-email send limits, failed-password throttling per email and per IP; JWTs are signed with a key derived
from SECRET_KEY, `alg` is pinned (so `alg: none` and key-confusion attacks fail), `iss`/`aud`/`exp`/`iat`/`jti` are
required, a challenge token can never be used as a session and vice versa; sessions are revocable; cookie is
HttpOnly + Secure + SameSite=Lax; Origin check on state-changing requests.
"""
import hashlib
import hmac
import logging
import re
import secrets
import uuid
from datetime import timedelta

import jwt as pyjwt

from django.conf import settings
from django.contrib.auth.hashers import check_password, make_password
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import transaction
from django.http import JsonResponse
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import mailer
from .api_errors import ApiError
from .models import Account, AppSettings, AuthSession, Conversation, LoginChallenge, QueryMetric, UserSource, VideoSource

log = logging.getLogger("studio.auth")

SESSION_COOKIE = "janani_session"
SESSION_DAYS = 30
OTP_TTL_MIN = 10
OTP_MAX_ATTEMPTS = 5
RESEND_COOLDOWN_S = 45
MAX_SENDS_PER_CHALLENGE = 5
MAX_CHALLENGES_PER_HOUR = 6
FAIL_LIMIT = 8                      # wrong passwords per email / per IP within FAIL_WINDOW_S
FAIL_WINDOW_S = 15 * 60
MIN_PASSWORD = 8

JWT_ALG = "HS256"
JWT_ISS = "janani"
AUD_OTP = "janani:otp"
AUD_SESSION = "janani:session"

OPEN_PATHS = {"/api/health/", "/api/auth/start/", "/api/auth/verify/", "/api/auth/resend/", "/api/auth/me/",
              "/api/auth/logout/"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


# ------------------------------------------------------------------ helpers

def _body(request):
    d = request.data
    if not isinstance(d, dict):
        raise ApiError("Expected a JSON object.")
    return d


def client_ip(request):
    fwd = request.META.get("HTTP_X_FORWARDED_FOR", "")
    return (fwd.split(",")[0].strip() or request.META.get("REMOTE_ADDR", ""))[:64]


def normalize_email(value):
    email = str(value or "").strip().lower()
    if not email or len(email) > 254:
        raise ApiError("Enter a valid email address.", code="invalid_email")
    try:
        validate_email(email)
    except ValidationError:
        raise ApiError("Enter a valid email address.", code="invalid_email") from None
    return email


def mask_email(email):
    local, _, domain = email.partition("@")
    return f"{local[:1]}{'•' * max(1, min(len(local) - 1, 6))}@{domain}"


def check_allowed(email):
    allow = settings.ALLOWED_EMAILS
    if allow and email not in allow and email.rsplit("@", 1)[-1] not in settings.ALLOWED_EMAIL_DOMAINS:
        raise ApiError("This workspace is invite-only. Ask the owner to add your email.", status=403, code="not_allowed")


def validate_new_password(password):
    if len(password) < MIN_PASSWORD:
        raise ApiError(f"Use at least {MIN_PASSWORD} characters for your password.", code="weak_password")
    if len(password) > 200:
        raise ApiError("That password is too long.", code="weak_password")
    if password.isdigit() or password.lower() in {"password", "password1", "12345678", "qwertyui", "iloveyou"}:
        raise ApiError("Choose a less guessable password.", code="weak_password")


def _hash_code(challenge_id, code):
    return hmac.new(settings.SECRET_KEY.encode(), f"{challenge_id}:{code}".encode(), hashlib.sha256).hexdigest()


def _hash_token(value):
    return hashlib.sha256(value.encode()).hexdigest()


class JwtInvalid(Exception):
    """Bad signature, wrong audience/issuer, malformed, missing claims, or an unsupported algorithm."""


class JwtExpired(Exception):
    pass


def _jwt_key():
    # a dedicated key derived from SECRET_KEY (>= 32 bytes) so the JWT key differs from the code-HMAC key
    return hashlib.sha256(b"janani-jwt-v1:" + settings.SECRET_KEY.encode()).digest()


def _jwt_encode(claims, audience, expires_at):
    now = timezone.now()
    payload = {**claims, "iss": JWT_ISS, "aud": audience, "iat": int(now.timestamp()), "exp": int(expires_at.timestamp())}
    return pyjwt.encode(payload, _jwt_key(), algorithm=JWT_ALG)


def _jwt_decode(token, audience, verify_exp=True):
    """Verified claims, or raises JwtExpired / JwtInvalid. The algorithm is pinned to HS256."""
    if not token or not isinstance(token, str) or token.count(".") != 2 or len(token) > 4096:
        raise JwtInvalid("malformed")
    try:
        return pyjwt.decode(token, _jwt_key(), algorithms=[JWT_ALG], audience=audience, issuer=JWT_ISS,
                            options={"require": ["exp", "iat", "jti", "sub", "iss", "aud"], "verify_exp": verify_exp})
    except pyjwt.ExpiredSignatureError:
        raise JwtExpired("expired") from None
    except pyjwt.PyJWTError as exc:
        raise JwtInvalid(type(exc).__name__) from None


def challenge_token(ch):
    """JWT for a pending one-time code. Built purely from the stored row, so it can be re-issued after a cooldown."""
    return _jwt_encode({"sub": ch.email, "jti": str(ch.id), "pur": ch.purpose, "otp": ch.code_hash}, AUD_OTP, ch.expires_at)


def create_session_token(account, user_agent=""):
    """Creates the revocable server-side session row and returns the session JWT."""
    jti = uuid.uuid4().hex
    expires = timezone.now() + timedelta(days=SESSION_DAYS)
    AuthSession.objects.create(account=account, token_hash=_hash_token(jti), user_agent=user_agent[:200], expires_at=expires)
    return _jwt_encode({"sub": str(account.id), "email": account.email, "jti": jti}, AUD_SESSION, expires)


def _bearer_or_cookie(request):
    auth = request.META.get("HTTP_AUTHORIZATION", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return request.COOKIES.get(SESSION_COOKIE)


def session_from_token(token):
    """AuthSession for a valid, unrevoked session JWT, else None."""
    try:
        claims = _jwt_decode(token, AUD_SESSION)
    except (JwtInvalid, JwtExpired):
        return None
    return (AuthSession.objects.select_related("account")
            .filter(token_hash=_hash_token(claims["jti"]), account_id=claims["sub"], expires_at__gt=timezone.now(),
                    account__is_active=True).first())


def _new_code():
    return f"{secrets.randbelow(10 ** 6):06d}"


_dummy_hash = None


def _burn_password_check(password):
    """Same cost as a real password check so response time does not reveal whether an email exists."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = make_password("janani-dummy-password")
    check_password(password, _dummy_hash)


def _fail_keys(email, ip):
    return f"authfail:e:{email}", f"authfail:i:{ip}"


def _check_failures(email, ip):
    for key in _fail_keys(email, ip):
        if cache.get(key, 0) >= FAIL_LIMIT:
            raise ApiError("Too many failed attempts. Please wait a few minutes and try again.", status=429,
                           code="rate_limited", extra={"retry_after": FAIL_WINDOW_S})


def _note_failure(email, ip):
    for key in _fail_keys(email, ip):
        cache.add(key, 0, FAIL_WINDOW_S)
        try:
            cache.incr(key)
        except ValueError:
            cache.set(key, 1, FAIL_WINDOW_S)


def _clear_failures(email, ip):
    for key in _fail_keys(email, ip):
        cache.delete(key)


def _send_code(email, code, purpose):
    try:
        mailer.send_otp(email, code, purpose, OTP_TTL_MIN)
    except mailer.EmailNotConfigured as exc:
        log.error("Email sign-in is not configured: %s", exc)
        raise ApiError("Email sign-in isn't set up on this server yet. The owner needs to configure the email service.",
                       status=503, code="email_not_configured") from None
    except mailer.EmailSendError as exc:
        log.error("Could not send login code to %s: %s | hint: %s", mask_email(email), exc, exc.hint)
        raise ApiError("We couldn't send the code right now. Please try again in a minute.", status=503,
                       code="email_failed") from None


def adopt_orphans(account):
    """The owner's account inherits everything created before sign-in existed (chats, videos, settings)."""
    Conversation.objects.filter(account__isnull=True).update(account=account)
    QueryMetric.objects.filter(account__isnull=True).update(account=account)
    AppSettings.objects.filter(pk=1, account__isnull=True).update(account=account)
    UserSource.objects.bulk_create([UserSource(account=account, source=s) for s in VideoSource.objects.all()],
                                   ignore_conflicts=True)


def _set_cookie(response, token):
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_DAYS * 86400, httponly=True,
                        secure=not settings.DEBUG and not settings.TESTING, samesite="Lax", path="/")


def _account_dict(a):
    return {"email": a.email, "created": a.created.isoformat()}


# ------------------------------------------------------------------ middleware

class AuthMiddleware:
    """Everything under /api/ needs a valid session except sign-in itself and /api/health/. Also blocks
    cross-site state-changing requests (CSRF defence in depth on top of SameSite=Lax)."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path
        request.account = None
        if not path.startswith("/api/"):
            return self.get_response(request)

        if request.method not in SAFE_METHODS:
            origin = request.META.get("HTTP_ORIGIN")
            if origin and not self._same_site(request, origin):
                return JsonResponse({"error": {"code": "bad_origin", "message": "Cross-site request blocked."}}, status=403)

        sess = session_from_token(_bearer_or_cookie(request))
        if sess:
            request.account = sess.account
            if timezone.now() - sess.last_used > timedelta(hours=1):
                AuthSession.objects.filter(pk=sess.pk).update(last_used=timezone.now())

        if request.account is None and path not in OPEN_PATHS:
            return JsonResponse({"error": {"code": "auth_required", "message": "Please sign in to continue."}}, status=401)
        return self.get_response(request)

    @staticmethod
    def _same_site(request, origin):
        from urllib.parse import urlparse
        parsed = urlparse(origin)
        host = parsed.netloc.lower()
        if host == request.get_host().lower() or origin in settings.CORS_ALLOWED_ORIGINS:
            return True
        if host == request.META.get("HTTP_X_FORWARDED_HOST", "").split(",")[0].strip().lower() and host:
            return True                                   # behind a reverse proxy that forwards the public host
        # Local development: the Vite dev server proxies /api and rewrites Host, so allow localhost origins - only in DEBUG.
        return settings.DEBUG and (parsed.hostname or "") in {"localhost", "127.0.0.1", "::1"}


# ------------------------------------------------------------------ views

@api_view(["POST"])
def auth_start(request):
    d = _body(request)
    email = normalize_email(d.get("email"))
    password = str(d.get("password") or "")
    ip = client_ip(request)
    if not password:
        raise ApiError("Enter your password.", code="missing_password")
    if len(password) > 200:
        raise ApiError("Invalid email or password.", status=401, code="invalid_credentials")
    check_allowed(email)
    _check_failures(email, ip)

    account = Account.objects.filter(email=email).first()
    if account:
        if not (account.is_active and check_password(password, account.password_hash)):
            _note_failure(email, ip)
            raise ApiError("Invalid email or password.", status=401, code="invalid_credentials")
        purpose, password_hash = "login", ""
    else:
        _burn_password_check(password)
        validate_new_password(password)
        purpose, password_hash = "signup", make_password(password)

    now = timezone.now()
    if LoginChallenge.objects.filter(email=email, created__gte=now - timedelta(hours=1)).count() >= MAX_CHALLENGES_PER_HOUR:
        raise ApiError("Too many codes requested for this email. Please try again later.", status=429, code="rate_limited",
                       extra={"retry_after": 3600})
    latest = LoginChallenge.objects.filter(email=email, consumed_at__isnull=True, expires_at__gt=now).order_by("-created").first()
    if latest and (now - latest.last_sent).total_seconds() < RESEND_COOLDOWN_S:
        wait = int(RESEND_COOLDOWN_S - (now - latest.last_sent).total_seconds()) + 1
        raise ApiError(f"A code was just sent. You can request another in {wait} seconds.", status=429, code="cooldown",
                       extra={"retry_after": wait, "challenge_token": challenge_token(latest)})

    code = _new_code()
    cid = uuid.uuid4()
    ch = LoginChallenge(id=cid, email=email, purpose=purpose, code_hash=_hash_code(cid, code), password_hash=password_hash,
                        expires_at=now + timedelta(minutes=OTP_TTL_MIN), ip=ip)
    ch.save()
    try:
        _send_code(email, code, purpose)
    except ApiError:
        ch.delete()
        raise
    LoginChallenge.objects.filter(email=email, consumed_at__isnull=True).exclude(pk=cid).update(consumed_at=now)
    return Response({"challenge_token": challenge_token(ch), "email": email, "masked": mask_email(email), "purpose": purpose,
                     "expires_in": OTP_TTL_MIN * 60, "resend_in": RESEND_COOLDOWN_S})


@api_view(["POST"])
def auth_verify(request):
    d = _body(request)
    code = re.sub(r"\D", "", str(d.get("code") or ""))
    if len(code) != 6:
        raise ApiError("Enter the 6-digit code.", code="invalid_code")
    try:
        claims = _jwt_decode(d.get("challenge_token"), AUD_OTP)
        cid = uuid.UUID(str(claims["jti"]))
    except JwtExpired:
        raise ApiError("This code has expired. Request a new one.", code="code_expired") from None
    except (JwtInvalid, ValueError):
        raise ApiError("This code is no longer valid. Request a new one.", code="challenge_invalid") from None

    now = timezone.now()
    outcome = None
    with transaction.atomic():
        ch = LoginChallenge.objects.select_for_update().filter(pk=cid).first()
        if not ch or ch.consumed_at or ch.email != claims["sub"] or not hmac.compare_digest(ch.code_hash, str(claims["otp"])):
            # unknown, already used, or superseded by a newer code (resend) - the old JWT is dead
            outcome = ("challenge_invalid", "This code is no longer valid. Request a new one.", 400, {})
        elif ch.expires_at <= now:
            outcome = ("code_expired", "This code has expired. Request a new one.", 400, {})
        elif ch.attempts >= OTP_MAX_ATTEMPTS:
            outcome = ("too_many_attempts", "Too many wrong codes. Request a new one.", 429, {})
        else:
            ch.attempts += 1
            if hmac.compare_digest(_hash_code(ch.id, code), str(claims["otp"])):
                ch.consumed_at = now
                ch.save(update_fields=["attempts", "consumed_at"])
            else:
                left = OTP_MAX_ATTEMPTS - ch.attempts
                if left <= 0:
                    ch.consumed_at = now
                ch.save(update_fields=["attempts", "consumed_at"])
                outcome = ("wrong_code", "That code is not correct." + (f" {left} attempt{'s' if left != 1 else ''} left." if left else
                           " Request a new code."), 400, {"attempts_left": left})
    if outcome:                                      # raised outside the transaction so the attempt counter is kept
        raise ApiError(outcome[1], status=outcome[2], code=outcome[0], extra=outcome[3])

    with transaction.atomic():
        if ch.purpose == "signup":
            account, created = Account.objects.get_or_create(email=ch.email, defaults={"password_hash": ch.password_hash})
            if created and account.email == settings.OWNER_EMAIL:
                adopt_orphans(account)
        else:
            account = Account.objects.filter(email=ch.email, is_active=True).first()
            if account is None:
                raise ApiError("This account is not available.", status=403, code="inactive")
        account.last_login = now
        account.save(update_fields=["last_login"])
    _clear_failures(ch.email, client_ip(request))
    token = create_session_token(account, request.META.get("HTTP_USER_AGENT", ""))
    resp = Response({"account": _account_dict(account), "token_type": "Bearer", "expires_in": SESSION_DAYS * 86400})
    _set_cookie(resp, token)
    return resp


@api_view(["POST"])
def auth_resend(request):
    d = _body(request)
    try:
        claims = _jwt_decode(d.get("challenge_token"), AUD_OTP, verify_exp=False)    # an expired code may still be re-sent
        cid = uuid.UUID(str(claims["jti"]))
    except (JwtInvalid, JwtExpired, ValueError):
        raise ApiError("This code is no longer valid. Start again.", code="challenge_invalid") from None
    now = timezone.now()
    ch = LoginChallenge.objects.filter(pk=cid, email=claims["sub"], consumed_at__isnull=True).first()
    if not ch:
        raise ApiError("This sign-in has expired. Start again.", code="challenge_invalid")
    wait = RESEND_COOLDOWN_S - (now - ch.last_sent).total_seconds()
    if wait > 0:
        raise ApiError(f"Please wait {int(wait) + 1} seconds before requesting another code.", status=429, code="cooldown",
                       extra={"retry_after": int(wait) + 1})
    if ch.sends >= MAX_SENDS_PER_CHALLENGE:
        raise ApiError("Too many codes sent. Start again in a little while.", status=429, code="rate_limited")
    code = _new_code()
    _send_code(ch.email, code, ch.purpose)
    ch.code_hash = _hash_code(ch.id, code)
    ch.attempts, ch.sends, ch.last_sent = 0, ch.sends + 1, now
    ch.expires_at = now + timedelta(minutes=OTP_TTL_MIN)
    ch.save(update_fields=["code_hash", "attempts", "sends", "last_sent", "expires_at"])
    return Response({"challenge_token": challenge_token(ch), "masked": mask_email(ch.email), "expires_in": OTP_TTL_MIN * 60,
                     "resend_in": RESEND_COOLDOWN_S})


@api_view(["GET"])
def auth_me(request):
    if request.account is None:
        raise ApiError("Please sign in to continue.", status=401, code="auth_required")
    return Response({"account": _account_dict(request.account)})


@api_view(["POST"])
def auth_logout(request):
    sess = session_from_token(_bearer_or_cookie(request))
    if sess:
        sess.delete()                                  # revokes the JWT's jti: it is dead even though its signature is valid
    resp = Response(status=204)
    resp.delete_cookie(SESSION_COOKIE, path="/", samesite="Lax")
    return resp
