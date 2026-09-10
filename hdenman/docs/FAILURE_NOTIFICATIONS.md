# Failure notifications for `run_stack_handler`

## Problem

`run_stack_handler` is a long-lived trading process. If it crashes after cron starts it, cron will not restart it until the next scheduled run. The repo already has critical-log email support, but the current crash path is not robust enough by itself:

- production email logging only exists when `PYSYS_LOGGING_CONFIG=syslogging.logging_prod.yaml` is active;
- `processToRun.run_process()` can log a critical failure, but currently only logs a short message;
- the email handler sends `record.msg`, not the formatted record with traceback;
- Python-level logging cannot catch interpreter startup failures, `SIGKILL`, OOM kills, or logging/email misconfiguration.

## Recommended design

Use two independent layers:

1. **In-process fatal logging**: central Python exception handling logs a critical record with traceback and sends an email.
2. **Supervisor-level failure alerting**: `systemd` detects process death, restarts the service, and runs a separate failure-email unit.

This gives useful Python diagnostics when Python is alive, plus a second alert/restart path when it is not.

## Existing repo behavior

### Process crash hook

`syscontrol/run_process.py` already catches unhandled exceptions around the main process loop:

```python
except Exception:
    config = self.data.config
    if config.get_element_or_default("log_failed_processes", False):
        self.log.critical(f"Process {self.process_name} failed!")
    raise
```

If `log_failed_processes: True`, this emits a `CRITICAL` log record.

Limitations:

- no exception type;
- no exception message;
- no traceback in the email body;
- process still exits, which is correct, but cron will not restart it.

### Production email handler

`syslogging/logging_prod.yaml` configures an email handler for critical records:

```yaml
handlers:
  email:
    class: syslogging.handlers.PstSMTPHandler
    level: CRITICAL
    formatter: simple
```

`syslogging.handlers.PstSMTPHandler` sends one SMTP email per critical log record.

Current implementation:

```python
def emit(self, record):
    try:
        subject_line = f"*{record.levelname}*: {record.msg}"
        send_mail_msg(record.msg, subject_line)
    except Exception as exc:
        print(f"Problem sending message: {exc}")
```

Limitations:

- uses `record.msg`, not `record.getMessage()`;
- does not use `self.format(record)`;
- therefore does not include formatted context or traceback from `exc_info=True`;
- email send failure is printed but not durably queued by this handler.

### Runtime config requirement

Production logging is only enabled when this environment variable is set:

```sh
PYSYS_LOGGING_CONFIG=syslogging.logging_prod.yaml
```

Without it, `syslogging.logger.get_logger()` falls back to sim logging, which writes to console only.

`devenv.nix` should set this for production commands, or the `run_stack_handler` service/cron command should set it explicitly.

## Layer 1: improve Python crash emails

### Enable fatal-process logging

Set this in private production config:

```yaml
log_failed_processes: True
```

### Enable production logging in runtime

Preferred in `devenv.nix`:

```nix
env.PYSYS_LOGGING_CONFIG = "syslogging.logging_prod.yaml";
```

Alternative in a service or cron command:

```sh
PYSYS_LOGGING_CONFIG=syslogging.logging_prod.yaml python -m sysproduction.run_stack_handler
```

### Improve `processToRun.run_process()`

Change the crash log to include traceback:

```python
except Exception:
    config = self.data.config
    if config.get_element_or_default("log_failed_processes", False):
        self.log.critical(
            "Process %s failed with unhandled exception",
            self.process_name,
            exc_info=True,
        )
    raise
```

Keep the re-raise. The process should still fail fast after an unexpected exception; the supervisor should restart or alert.

### Improve `PstSMTPHandler.emit()`

Change the email body to the formatted record and the subject to the interpolated message:

```python
def emit(self, record):
    try:
        subject_line = f"*{record.levelname}*: {record.getMessage()}"
        body = self.format(record)
        send_mail_msg(body, subject_line)
    except Exception as exc:
        print(f"Problem sending message: {exc}")
```

Result:

- subject stays short;
- body includes timestamp, logger name, level, message, and traceback when present;
- every `processToRun` process benefits, not only `run_stack_handler`.

## Layer 2: supervise `run_stack_handler` with systemd

Cron is acceptable for short batch jobs. It is weak for a long-lived trading daemon because it does not restart a mid-day crash.

Run `run_stack_handler` as a `systemd` service instead:

```ini
[Unit]
Description=pysystemtrade run_stack_handler
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/hdenman/algo-trading/pysystemtrade
Environment=PYSYS_LOGGING_CONFIG=syslogging.logging_prod.yaml
Environment=PYSYS_PRIVATE_CONFIG_DIR=/home/hdenman/algo-trading/pysystemtrade_config
ExecStart=/path/to/devenv shell -- python -m sysproduction.run_stack_handler
Restart=on-failure
RestartSec=30
OnFailure=pysystemtrade-failure-email@%n.service
User=hdenman

[Install]
WantedBy=multi-user.target
```

Then add a failure notification unit:

```ini
[Unit]
Description=Email pysystemtrade service failure for %i

[Service]
Type=oneshot
WorkingDirectory=/home/hdenman/algo-trading/pysystemtrade
Environment=PYSYS_PRIVATE_CONFIG_DIR=/home/hdenman/algo-trading/pysystemtrade_config
ExecStart=/path/to/devenv shell -- python -m sysproduction.email_process_failure %i
User=hdenman
```

The notifier module should be small and dependency-light:

- accept service/process name from argv;
- include hostname;
- include timestamp;
- include failed systemd unit name;
- include recent journal lines if practical;
- include path to the relevant echo/log file when known;
- send via `syslogdiag.emailing.send_mail_msg()`;
- exit non-zero if email fails, so `journalctl -u pysystemtrade-failure-email@...` shows the failure.

## Suggested `sysproduction.email_process_failure` behavior

Pseudo-code:

```python
import socket
import subprocess
import sys
import datetime

from syslogdiag.emailing import send_mail_msg


def main():
    unit_name = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    hostname = socket.gethostname()
    now = datetime.datetime.now().isoformat(timespec="seconds")

    journal = subprocess.run(
        ["journalctl", "-u", unit_name, "-n", "80", "--no-pager"],
        text=True,
        capture_output=True,
        check=False,
    )

    body = f"""
Pysystemtrade service failure

Host: {hostname}
Unit: {unit_name}
Time: {now}

Recent journal:
{journal.stdout or journal.stderr}
""".strip()

    subject = f"pysystemtrade failure: {unit_name} on {hostname}"
    send_mail_msg(body, subject)


if __name__ == "__main__":
    main()
```

This script should not use the normal logging system for its primary notification path; if logging is broken, the failure notifier still needs to work.

## Optional narrower configuration

If emailing for every failed process is too noisy, replace the boolean-only behavior with a compatibility helper:

```yaml
log_failed_processes:
  - run_stack_handler
```

Compatibility behavior:

- `False`: no process-failure emails;
- `True`: all `processToRun` processes email on failure;
- list of names: only listed process names email on failure.

This avoids duplicate/noisy alerts while preserving the current default.

## Verification plan

### App-level email path

1. Configure production logging and SMTP settings.
2. Set `log_failed_processes: True`.
3. Add a temporary test process or monkeypatch a timer method to raise `RuntimeError("notification smoke test")`.
4. Run the process.
5. Confirm:
   - process exits non-zero;
   - echo file contains traceback;
   - email arrives;
   - email body includes traceback;
   - subject includes process name.

### Systemd path

1. Start the `run_stack_handler` service.
2. Force a controlled failure with a temporary test process or `systemctl kill --signal=SIGTERM` / a synthetic failing unit.
3. Confirm:
   - `systemd` marks failure for a real crash;
   - `OnFailure=` unit runs;
   - failure email arrives;
   - `Restart=on-failure` restarts the main service where appropriate;
   - journal records both the original failure and notification result.

Avoid testing with `SIGKILL` first: it proves supervisor behavior but skips Python cleanup and can leave process-control state stale. Use a normal unhandled exception first, then test hard-kill behavior after the notification path works.

## Operational recommendation

Do not rely on moving IB Gateway restart time as the alerting mechanism. Move the gateway restart away from `09:30` because market open is a bad maintenance window, but still implement crash notification and supervision.

Best target state:

- `run_stack_handler` runs under `systemd`, not cron;
- production logging is enabled in its environment;
- `log_failed_processes` is enabled at least for `run_stack_handler`;
- Python crash emails include traceback;
- `systemd` restarts on failure and sends a separate failure email;
- Gateway restart is scheduled outside active trading/reconciliation windows.
