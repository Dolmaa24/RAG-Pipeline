#!/usr/bin/env bash
# Start the stack, or one piece of it.
#
#   ./run.sh              # everything: both workers, API, dashboard
#   ./run.sh worker-io    # I/O worker only (threads, high concurrency)
#   ./run.sh worker-cpu   # CPU/GPU worker only (Whisper, LLM, Chromium)
#   ./run.sh worker-agents # investigation worker only (threads, model calls)
#   ./run.sh worker-build  # build worker only (writes generated code)
#   ./run.sh mcp          # MCP server over stdio (add --http for the HTTP one)
#   ./run.sh api
#   ./run.sh dashboard
#   ./run.sh flower
#
# The two env vars set below are the macOS traps this script exists to defuse:
#
#   OBJC_DISABLE_INITIALIZE_FORK_SAFETY — Celery's prefork pool forks after the
#     Objective-C runtime has initialised, and the child then hangs the first
#     time it touches a framework (Vision, CoreML, MLX). It is also set in
#     celery_app.py, but exported here too so it covers anything started
#     before that import.
#
#   ulimit -n — the default of 256 open files on macOS is well under what 16
#     concurrent fetchers plus Chromium need.

set -euo pipefail
cd "$(dirname "$0")"

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=.
ulimit -n 4096 2>/dev/null || echo "note: could not raise the file-descriptor limit" >&2

VENV=./venv/bin
[ -x "$VENV/python" ] || { echo "No venv found. See README.md → Setup." >&2; exit 1; }

check_redis() {
  if ! redis-cli ping >/dev/null 2>&1; then
    echo "Redis is not responding on localhost:6379." >&2
    echo "Start it with:  brew services start redis    (or: docker compose up -d)" >&2
    exit 1
  fi
}

worker_io() {
  # Network-bound: the threads are almost always waiting, so concurrency is
  # cheap. The per-host rate limiter, not this number, is what protects sites.
  check_redis
  exec "$VENV/celery" -A celery_app worker \
    --queues=io --pool=threads --concurrency=16 \
    --hostname=io@%h --loglevel=info
}

worker_cpu() {
  # Whisper, the local model, Chromium and OCR each want a core or the GPU.
  # Keeping this narrow is the point: two of them at once is already contention.
  check_redis
  CELERY_WORKER_QUEUE=cpu exec "$VENV/celery" -A celery_app worker \
    --queues=cpu --pool=prefork --concurrency=2 \
    --hostname=cpu@%h --loglevel=info
}

worker_agents() {
  # Threads, not prefork. An investigation holds model clients and a thread
  # pool, which is exactly the shape that trips the prefork pool's four-second
  # startup handshake and macOS's refusal to let Metal survive fork(). Both are
  # already scars in this codebase. Concurrency is low because each run is
  # minutes of model calls, and two at once on 8 GB is contention.
  check_redis
  exec "$VENV/celery" -A celery_app worker \
    --queues=agents --pool=threads --concurrency=2 \
    --hostname=agents@%h --loglevel=info
}

worker_build() {
  # Its own queue, and therefore its own worker. A build is five or more
  # whole-file generations and runs for minutes; on the agents queue it would
  # sit in front of every investigation, which is the head-of-line blocking
  # that queue was split off to avoid in the first place.
  #
  # Concurrency 1: two builds at once means two model clients and two pytest
  # subprocesses on a machine that is already holding an embedder.
  check_redis
  exec "$VENV/celery" -A celery_app worker \
    --queues=build --pool=threads --concurrency=1 \
    --hostname=build@%h --loglevel=info
}

api()       { exec "$VENV/uvicorn" app:app --host 127.0.0.1 --port 8000 --reload; }
# Not part of `all`: an MCP client spawns its own copy over stdio, and a
# long-lived one is only wanted for the HTTP transport.
mcp()       { exec "$VENV/python" mcp_server.py "$@"; }
dashboard() { exec "$VENV/streamlit" run dashboard.py; }
flower()    { check_redis; exec "$VENV/celery" -A celery_app flower --port=5555; }

all() {
  check_redis
  trap 'kill 0' EXIT INT TERM   # one Ctrl-C stops the whole stack
  "$0" worker-io     & sleep 1
  "$0" worker-cpu    & sleep 1
  "$0" worker-agents & sleep 1
  "$0" worker-build  & sleep 1
  "$0" api        & sleep 2
  "$0" dashboard  &
  echo
  echo "  API        http://127.0.0.1:8000/docs"
  echo "  Dashboard  http://localhost:8501"
  echo "  Health     curl localhost:8000/health"
  echo
  echo "Ctrl-C stops everything."
  wait
}

case "${1:-all}" in
  worker-io)  worker_io ;;
  worker-cpu) worker_cpu ;;
  worker-agents) worker_agents ;;
  worker-build) worker_build ;;
  api)        api ;;
  dashboard)  dashboard ;;
  flower)     flower ;;
  mcp)        shift; mcp "$@" ;;
  all)        all ;;
  *) echo "usage: $0 [all|worker-io|worker-cpu|worker-agents|worker-build|api|dashboard|flower|mcp]" >&2; exit 2 ;;
esac
