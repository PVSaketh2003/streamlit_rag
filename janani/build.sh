#!/usr/bin/env bash
# Render build: Python deps + production React build (served by Django/WhiteNoise).
set -euo pipefail
python -m pip install --upgrade pip
pip install -r requirements.txt
npm ci --prefix frontend
npm run build --prefix frontend
