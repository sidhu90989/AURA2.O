#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip >/dev/null 2>&1 || true
pip install -r backend/requirements.txt

echo "Starting backend..."
uvicorn backend.app.main:app --reload --port 8000 &
BACKEND_PID=$!

pushd frontend >/dev/null
if [ ! -d node_modules ]; then
  npm install
fi
echo "Starting frontend..."
npm run dev &
FRONT_PID=$!
popd >/dev/null

trap 'echo "Stopping..."; kill $BACKEND_PID $FRONT_PID 2>/dev/null || true' INT TERM
wait $FRONT_PID
