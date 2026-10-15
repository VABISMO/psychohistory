#!/usr/bin/env bash
# Arranca backend (FastAPI) y frontend (Next.js) para Psychohistory
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$ROOT/.logs"
mkdir -p "$LOGDIR"

echo "=== Psychohistory launcher ==="

# --- Backend ---
pkill -f "uvicorn psychohistory.main" 2>/dev/null || true
sleep 1
cd "$ROOT/backend"
setsid nohup venv/bin/uvicorn psychohistory.main:app \
  --host 0.0.0.0 --port 8000 \
  > "$LOGDIR/backend.log" 2>&1 < /dev/null &
echo "Backend PID: $!  -> http://localhost:8000"

# --- Frontend ---
pkill -f "next dev" 2>/dev/null || true
sleep 1
cd "$ROOT/frontend"
setsid nohup npm run dev -- --port 3000 \
  > "$LOGDIR/frontend.log" 2>&1 < /dev/null &
echo "Frontend PID: $!  -> http://localhost:3000"

sleep 6
echo ""
echo "--- Backend check ---"
curl -s http://localhost:8000/ && echo "" || echo "backend aun no responde"
echo "--- Frontend check ---"
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000/ || echo "frontend aun no responde"
echo ""
echo "Logs en $LOGDIR/"
