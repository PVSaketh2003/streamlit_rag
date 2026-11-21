"""Sends the one-time login code.

Providers (chosen automatically, override with EMAIL_PROVIDER=smtp|brevo|console):
  * smtp    – Gmail (or any SMTP server). Needs EMAIL_HOST_PASSWORD (a Google *App Password*).
              NOTE: Render's FREE web services block outbound ports 25/465/587, so SMTP needs a paid Render instance.
  * brevo   – HTTPS API (api.brevo.com), works from any host incl. Render Free. Needs BREVO_API_KEY and the sender
              address verified in Brevo.
  * console – prints the code to the server log. Only allowed when DEBUG is on (local development) or in tests.
"""
import json
import logging
import smtplib
import socket
import urllib.error
import urllib.request
from html import escape

from django.conf import settings
from django.core.mail import EmailMultiAlternatives, get_connection

log = logging.getLogger("studio.mail")


class EmailNotConfigured(Exception):
    pass


class EmailSendError(Exception):
    """The provider refused or could not be reached. `hint` is safe to log, never shown to end users."""

    def __init__(self, message, hint=""):
        super().__init__(message)
        self.hint = hint


def provider():
    return settings.EMAIL_PROVIDER


def status():
    """Non-secret summary for /api/health/ so a broken mail setup is diagnosable."""
    p = provider()
    return {"provider": p, "configured": p in ("smtp", "brevo") or (p == "console" and (settings.DEBUG or settings.TESTING)),
            "from": settings.DEFAULT_FROM_EMAIL}


def _message(code, purpose, ttl_min):
    action = "finish creating your account" if purpose == "signup" else "sign in"
    subject = f"{code} is your Janani verification code"
    text = (f"Your Janani verification code is {code}\n\n"
            f"Enter it within {ttl_min} minutes to {action}.\n"
            "If you did not request this, you can ignore this email - nobody can sign in without the code.\n")
    digits = "".join(f'<span style="display:inline-block;width:42px;height:52px;line-height:52px;margin:0 3px;'
                     f'border-radius:12px;background:#f3f2fb;font:600 26px/52px Menlo,Consolas,monospace;'
                     f'color:#0e0f1a;text-align:center">{escape(ch)}</span>' for ch in code)
    html = f"""<!doctype html><html><body style="margin:0;background:#fafaf7;padding:32px 16px;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif">
<div style="max-width:460px;margin:0 auto;background:#ffffff;border:1px solid #ececf3;border-radius:22px;padding:32px;text-align:center">
<div style="font:700 20px/1 -apple-system,Segoe UI,Arial;letter-spacing:-.02em;color:#0e0f1a">Janani<span style="font-style:italic;font-weight:400;font-family:Georgia,serif"> studio</span></div>
<p style="color:#6a6e84;font-size:15px;margin:22px 0 18px">Your verification code to {escape(action)}</p>
<div style="margin:0 0 20px">{digits}</div>
<p style="color:#6a6e84;font-size:13px;margin:0">This code expires in {ttl_min} minutes.<br>If you did not request it, you can safely ignore this email.</p>
</div></body></html>"""
    return subject, text, html


def _send_brevo(to_email, subject, text, html):
    sender_email = settings.DEFAULT_FROM_EMAIL_ADDRESS
    payload = {"sender": {"name": "Janani", "email": sender_email}, "to": [{"email": to_email}],
               "subject": subject, "textContent": text, "htmlContent": html}
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email", data=json.dumps(payload).encode(), method="POST",
        headers={"api-key": settings.BREVO_API_KEY, "content-type": "application/json", "accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=settings.EMAIL_TIMEOUT) as resp:
            if resp.status >= 300:
                raise EmailSendError(f"Brevo HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        hint = "Brevo rejected the request: verify the sender address in Brevo and check BREVO_API_KEY."
        raise EmailSendError(f"Brevo HTTP {exc.code}: {body}", hint) from None
    except (urllib.error.URLError, socket.timeout, OSError) as exc:
        raise EmailSendError(f"Could not reach Brevo: {exc}", "Check the server's internet access.") from None


def send_otp(to_email, code, purpose="login", ttl_min=10):
    p = provider()
    subject, text, html = _message(code, purpose, ttl_min)
    if p == "brevo":
        if not settings.BREVO_API_KEY:
            raise EmailNotConfigured("BREVO_API_KEY is not set")
        return _send_brevo(to_email, subject, text, html)
    if p == "console":
        if not (settings.DEBUG or settings.TESTING):
            raise EmailNotConfigured("No email provider is configured (set EMAIL_HOST_PASSWORD or BREVO_API_KEY)")
        log.warning("DEV ONLY - verification code for %s: %s", to_email, code)
        if not settings.TESTING:
            return
    elif p == "smtp":
        if not settings.EMAIL_HOST_PASSWORD and not settings.TESTING:
            raise EmailNotConfigured("EMAIL_HOST_PASSWORD (a Google App Password) is not set")
    else:
        raise EmailNotConfigured(f"Unknown EMAIL_PROVIDER {p!r}")
    try:
        conn = get_connection(fail_silently=False, timeout=settings.EMAIL_TIMEOUT)
        msg = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [to_email], connection=conn)
        msg.attach_alternative(html, "text/html")
        msg.send()
    except smtplib.SMTPAuthenticationError:
        raise EmailSendError("SMTP login rejected", "Gmail refused the login. Use a Google App Password "
                             "(Google Account > Security > 2-Step Verification > App passwords), not the account password.") from None
    except smtplib.SMTPRecipientsRefused:
        raise EmailSendError("Recipient refused", "The mail server rejected the recipient address.") from None
    except (smtplib.SMTPException, OSError, socket.timeout) as exc:
        hint = ("Could not connect to the SMTP server. Render's FREE web services block outbound SMTP ports 25/465/587: "
                "upgrade the service to a paid instance, or set BREVO_API_KEY to send over HTTPS instead.")
        raise EmailSendError(f"SMTP failure: {type(exc).__name__}: {exc}", hint) from None
