#!/usr/bin/env bash
# Render entry point when the service's Root Directory is the repository root.
# The app lives in janani/; this just runs its start script from there.
set -euo pipefail
cd "$(dirname "$0")/janani"
exec ./start.sh
