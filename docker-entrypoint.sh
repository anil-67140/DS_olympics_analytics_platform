#!/bin/sh
set -e
case "${1:-api}" in
  api)
    export SERVICE=api PORT="${PORT:-8000}"
    exec uvicorn apps.api.main:app --host 0.0.0.0 --port "$PORT" ;;
  dashboard)
    export SERVICE=dashboard PORT="${PORT:-7860}"
    exec python -m apps.dashboard.app ;;
  pipeline)
    shift
    exec python -m olympics.cli run "$@" ;;
  *)
    exec "$@" ;;
esac
