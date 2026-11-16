#!/usr/bin/env bash
# Starts Django (:8000) and the React dev server (:5173). Ctrl+C stops both.
set -e
cd "$(dirname "$0")"
PY=${PY:-/Users/pvsairamsaketh/Desktop/langchain/venv/bin/python}
PGBIN=/opt/homebrew/opt/postgresql@15/bin
$PGBIN/pg_isready -q || { echo "PostgreSQL is not running: brew services start postgresql@15"; exit 1; }
$PGBIN/psql -lqt | cut -d'|' -f1 | grep -qw "${POSTGRES_DB:-rag_studio}" || $PGBIN/createdb "${POSTGRES_DB:-rag_studio}"
(cd backend && $PY manage.py migrate --noinput)
[ -d frontend/node_modules ] || (cd frontend && npm install)
trap 'kill 0' EXIT
(cd backend && $PY manage.py runserver 127.0.0.1:8000 --noreload) &
(cd frontend && npm run dev -- --host 127.0.0.1) &
wait
