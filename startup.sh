#!/bin/bash
set -uo pipefail

[ -f ~/.profile ] && . ~/.profile

: "${MONGO_DATA:?MONGO_DATA is not set}"
: "${LOG_PATH:?LOG_PATH is not set}"
: "${PYSYS_CODE:?PYSYS_CODE is not set}"

DATE=$(date +%Y-%m-%d)

# --- MongoDB ---
if pgrep -x mongod > /dev/null 2>&1; then
    echo "mongod already running (PID $(pgrep -x mongod))"
else
    echo "Starting mongod..."
    nohup mongod --dbpath "$MONGO_DATA" >> "$LOG_PATH/mongodb.${DATE}.log" 2>&1 &
    echo "mongod started (PID $!)"
fi

# --- Logging daemon ---
if pgrep -f "syslogging/server.py" > /dev/null 2>&1; then
    echo "Logging server already running (PID $(pgrep -f 'syslogging/server.py'))"
else
    echo "Starting logging server..."
    nohup python -u "$PYSYS_CODE/syslogging/server.py" --file "$LOG_PATH/pysystemtrade.log" \
        >> "$LOG_PATH/logging_server.${DATE}.log" 2>&1 &
    echo "Logging server started (PID $!)"
fi
