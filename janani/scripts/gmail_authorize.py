#!/usr/bin/env python3
"""One-time Google sign-in that lets Janani send the login codes through the Gmail API (HTTPS, works on Render Free).

Run on your own computer (it opens your browser):
    python3 scripts/gmail_authorize.py

Needs GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET in backend/.env (see README: "Gmail API setup"). It saves
GMAIL_REFRESH_TOKEN into backend/.env and never prints any secret.
"""
import http.server
import json
import re
import secrets
import socket
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / "backend" / ".env"
SCOPE = "https://www.googleapis.com/auth/gmail.send"


def read_env(path=ENV_PATH):
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.split(" #", 1)[0].strip().strip('"').strip("'")
    return out


def write_env_value(key, value, path=ENV_PATH):
    """Set KEY=value in the .env file (replacing an existing line, keeping everything else)."""
    lines = path.read_text().splitlines() if path.is_file() else []
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for i, line in enumerate(lines):
        if pat.match(line):
            lines[i] = f"{key}={value}"
            break
    else:
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o600)


def main():
    env = read_env()
    cid, secret = env.get("GMAIL_CLIENT_ID", ""), env.get("GMAIL_CLIENT_SECRET", "")
    if not cid or not secret:
        sys.exit("Add GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET to backend/.env first (README: Gmail API setup).")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    redirect = f"http://127.0.0.1:{port}/"
    state = secrets.token_urlsafe(16)
    result = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if q.get("state", [""])[0] != state:
                self.send_response(400); self.end_headers(); self.wfile.write(b"State mismatch"); return
            result["code"] = q.get("code", [""])[0]
            result["error"] = q.get("error", [""])[0]
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("<h2>Janani is connected to Gmail.</h2><p>You can close this tab and return to the terminal.</p>".encode())

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": cid, "redirect_uri": redirect, "response_type": "code", "scope": SCOPE,
        "access_type": "offline", "prompt": "consent", "state": state})
    print("Opening your browser to authorize Janani to SEND email from your Gmail account (no reading access)...")
    print("If it does not open, visit:\n" + url + "\n")
    webbrowser.open(url)
    for _ in range(600):
        if result:
            break
        threading.Event().wait(0.5)
    if not result.get("code"):
        sys.exit(f"Authorization failed: {result.get('error') or 'timed out'}")

    data = urllib.parse.urlencode({"code": result["code"], "client_id": cid, "client_secret": secret,
                                   "redirect_uri": redirect, "grant_type": "authorization_code"}).encode()
    req = urllib.request.Request("https://oauth2.googleapis.com/token", data=data, method="POST",
                                 headers={"content-type": "application/x-www-form-urlencoded"})
    try:
        tok = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as exc:
        sys.exit(f"Token exchange failed: {exc}")
    if not tok.get("refresh_token"):
        sys.exit("Google did not return a refresh token. Remove Janani at myaccount.google.com/permissions and run again.")
    write_env_value("GMAIL_REFRESH_TOKEN", tok["refresh_token"])
    print(f"Done. GMAIL_REFRESH_TOKEN was saved to {ENV_PATH} (not shown).")


if __name__ == "__main__":
    main()
