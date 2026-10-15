#!/usr/bin/env bash
# Psychohistory — PRODUCTION launcher.
# Builds the optimized frontend and serves backend + frontend with no dev tooling.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$ROOT/.logs"
mkdir -p "$LOGDIR"

echo "=== Psychohistory PRODUCTION launcher ==="

# --- Build frontend ---
echo "Building frontend (optimized)..."
( cd "$ROOT/frontend" && npm run build )

# --- Backend ---
pkill -f "[u]vicorn psychohistory.main" 2>/dev/null || true
sleep 1
( cd "$ROOT/backend" && setsid nohup venv/bin/uvicorn psychohistory.main:app \
    --host 0.0.0.0 --port 8000 \
    > "$LOGDIR/backend.log" 2>&1 < /dev/null & )
echo "Backend  -> http://localhost:8000"

# --- Frontend (production server) ---
pkill -f "[n]ext start" 2>/dev/null || true
pkill -f "[n]ext-server" 2>/dev/null || true
sleep 1
( cd "$ROOT/frontend" && setsid nohup ./node_modules/.bin/next start --port 3000 \
    > "$LOGDIR/frontend.log" 2>&1 < /dev/null & )
echo "Frontend -> http://localhost:3000"

sleep 5
echo ""
echo "--- Health check ---"
curl -s --max-time 6 -o /dev/null -w "backend   HTTP %{http_code}\n" http://localhost:8000/ || echo "backend   not responding yet"
curl -s --max-time 6 -o /dev/null -w "frontend  HTTP %{http_code}\n" http://localhost:3000/ || echo "frontend  not responding yet"
echo ""
echo "Logs in $LOGDIR/"
