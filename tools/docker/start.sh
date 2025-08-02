#!/usr/bin/env bash
set -euo pipefail

# Simple process supervisor for web-ui (FastAPI) and poller in one container
# - forwards SIGINT/SIGTERM to children
# - exits with non-zero if any child fails
# - prints tagged logs

APP_ROOT="/app"
WEB_DIR="${APP_ROOT}/web-ui/backend"
POLL_DIR="${APP_ROOT}/web-ui"

log() { echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')] [$1] $2"; }

# Signal handling
pids=()
term_children() {
  log "INIT" "Forwarding signals to children: ${pids[*]:-}"
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" || true
    fi
  done
}
trap term_children SIGINT SIGTERM

# Optionally activate venvs if present
activate_if_exists() {
  local venv_path="$1"
  if [[ -f "$venv_path" ]]; then
    # shellcheck disable=SC1090
    . "$venv_path"
  fi
}

# Start Web-UI
start_web() {
  cd "$WEB_DIR"
  activate_if_exists "${APP_ROOT}/venv/bin/activate"
  export PYTHONUNBUFFERED=1
  export DEPLOY_MODE="${DEPLOY_MODE:-docker}"
  exec python main.py
}

# Start Poller
start_poller() {
  cd "$POLL_DIR"
  activate_if_exists "${POLL_DIR}/venv/bin/activate"
  export PYTHONUNBUFFERED=1
  export DEPLOY_MODE="${DEPLOY_MODE:-docker}"
  exec python Services/zammad_rag_poller.py
}

log "INIT" "Starting web-ui and poller"
start_web & pids+=($!)
start_poller & pids+=($!)

# Wait for children, capture first failure
exit_code=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    ec=$?
    log "INIT" "Child process $pid exited with $ec"
    exit_code=$ec
    # terminate others
    term_children
  fi
done

log "INIT" "All child processes exited with code $exit_code"
exit "$exit_code"