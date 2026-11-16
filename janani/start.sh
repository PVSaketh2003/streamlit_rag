#!/usr/bin/env bash
# Render start: apply migrations, then serve API + React app with gunicorn.
# gthread workers keep Server-Sent Events (live streaming answers) flowing without blocking other requests.
set -euo pipefail
cd backend
python manage.py migrate --noinput
exec gunicorn config.wsgi:application \
  --bind "0.0.0.0:${PORT:-10000}" \
  --worker-class gthread --workers "${WEB_CONCURRENCY:-1}" --threads "${GUNICORN_THREADS:-8}" \
  --timeout 180 --graceful-timeout 30 --keep-alive 5 \
  --access-logfile - --error-logfile -
