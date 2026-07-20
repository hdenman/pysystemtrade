#!/bin/bash
set -uo pipefail

[ -f ~/.profile ] && . ~/.profile

: "${MONGO_DATA:?MONGO_DATA is not set}"
: "${LOG_PATH:?LOG_PATH is not set}"
: "${PYSYS_CODE:?PYSYS_CODE is not set}"

UNIVERSE="${PYSYS_UNIVERSE:-synthetic}"
MONGO_DATA_U="${MONGO_DATA%/}/${UNIVERSE}"
LOG_PATH_U="${LOG_PATH%/}/${UNIVERSE}"
mkdir -p "$MONGO_DATA_U" "$LOG_PATH_U"

DATE=$(date +%Y-%m-%d)

# --- MongoDB ---
MONGO_LOG="${LOG_PATH_U}/mongodb.${DATE}.log"
if pgrep -x mongod > /dev/null 2>&1; then
    # Already running — find the log file it's actually writing to
    existing=$(ls -t "${LOG_PATH_U}"/mongodb.*.log 2>/dev/null | head -1)
    echo "mongod already running (PID $(pgrep -x mongod))"
    echo "  Log: ${existing:-${MONGO_LOG} (not yet created)}"
else
    echo "Starting mongod..."
    nohup mongod --dbpath "$MONGO_DATA_U" >> "$MONGO_LOG" 2>&1 &
    echo "mongod started (PID $!)"
    echo "  Log: ${MONGO_LOG}"
fi

# --- Logging daemon ---
LOGGER_LOG="${LOG_PATH_U}/pysystemtrade.log"
LOGGER_STDERR="${LOG_PATH_U}/logging_server.${DATE}.log"
if pgrep -f "syslogging/server.py" > /dev/null 2>&1; then
    SERVER_PID=$(pgrep -f "syslogging/server.py")
    # Parse --file arg from the running process command line
    existing_log=$(ps -o args= -p "$SERVER_PID" 2>/dev/null \
        | sed -n 's/.*--file[[:space:]]\{1,\}\([^[:space:]]*\).*/\1/p')
    existing_stderr=$(ls -t "${LOG_PATH_U}"/logging_server.*.log 2>/dev/null | head -1)
    echo "Logging server already running (PID ${SERVER_PID})"
    echo "  Log: ${existing_log:-${LOGGER_LOG} (unknown)}"
    echo "  Stderr: ${existing_stderr:-unknown}"
else
    echo "Starting logging server..."
    nohup python -u "$PYSYS_CODE/syslogging/server.py" --file "$LOGGER_LOG" \
        >> "$LOGGER_STDERR" 2>&1 &
    echo "Logging server started (PID $!)"
    echo "  Log: ${LOGGER_LOG}"
    echo "  Stderr: ${LOGGER_STDERR}"
fi
