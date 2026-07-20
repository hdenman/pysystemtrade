#!/bin/bash
set -uo pipefail

[ -f ~/.profile ] && . ~/.profile

_wait_gone() {
    local match="$1" kind="$2" secs="${3:-10}"
    local i
    for i in $(seq 1 "$secs"); do
        pgrep -f "$match" > /dev/null 2>&1 || return 0
        sleep 1
    done
    echo "  Warning: $kind did not stop after ${secs}s — sending SIGKILL..."
    pkill -9 -f "$match" 2>/dev/null || true
}

# --- Logging daemon (stop first; it depends on MongoDB) ---
if pgrep -f "syslogging/server.py" > /dev/null 2>&1; then
    PID=$(pgrep -f "syslogging/server.py")
    echo "Stopping logging server (PID ${PID})..."
    kill "$PID" 2>/dev/null || true
    _wait_gone "syslogging/server.py" "logging server" 5
    echo "  Logging server stopped."
else
    echo "Logging server not running."
fi

# --- MongoDB ---
if pgrep -x mongod > /dev/null 2>&1; then
    PID=$(pgrep -x mongod)
    echo "Stopping mongod (PID ${PID})..."
    kill "$PID" 2>/dev/null || true
    _wait_gone "mongod" "mongod" 15
    echo "  mongod stopped."
else
    echo "mongod not running."
fi
