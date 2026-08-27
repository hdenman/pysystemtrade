"""
List the status of all production processes for the active universe.

Shows every process recorded in the MongoDB process_control collection:
run state (running / not running / crashed), admin status (GO / STOP / NO-RUN /
PAUSE), PID, last start/end times, and the configured time window from
control_config.yaml.

Run as a module::

    PYSYS_UNIVERSE=futures python -m util.list_processes
    PYSYS_UNIVERSE=futures python -m util.list_processes --methods  # per-method detail
"""

import argparse
import sys

import pandas as pd

from sysdata.data_blob import dataBlob
from sysproduction.data.control_process import dataControlProcess, diagControlProcess
from syscore.universe import get_universe

_DATE_FMT = "%Y-%m-%d %H:%M"


def _fmt_dt(dt) -> str:
    if dt is None:
        return "—"
    try:
        return dt.strftime(_DATE_FMT)
    except Exception:
        return str(dt)


def report_processes(show_methods: bool = False) -> pd.DataFrame:
    universe = get_universe().value
    print(f"\nUniverse : {universe}")

    with dataBlob(log_name="list_processes") as blob:
        ctrl   = dataControlProcess(blob)
        diag   = diagControlProcess(blob)

        process_names = sorted(diag.get_list_of_process_names())

        if not process_names:
            print("(no process records in database)")
            return pd.DataFrame()

        rows = []
        for name in process_names:
            cp = ctrl.db_control_process_data.get_control_for_process_name(name)
            start_cfg = diag.get_start_time(name).strftime("%H:%M")
            stop_cfg  = diag.get_stop_time(name).strftime("%H:%M")
            rows.append({
                "process":    name,
                "run_state":  cp.running_mode_str,
                "status":     cp.status,
                "pid":        cp.process_id if cp.currently_running else "—",
                "last_start": _fmt_dt(cp.last_start_time),
                "last_end":   _fmt_dt(cp.last_end_time),
                "window":     f"{start_cfg}–{stop_cfg}",
            })

        df = pd.DataFrame(rows)

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        pd.set_option("display.max_colwidth", 30)
        print(df.to_string(index=False))

        if show_methods:
            _print_method_detail(process_names, diag)

    return df


def _print_method_detail(process_names: list, diag: diagControlProcess) -> None:
    print("\n── Method detail ──────────────────────────────────────────────")
    for name in process_names:
        methods = diag.get_list_of_methods_for_process_name(name)
        if not methods:
            continue
        print(f"\n  {name}")
        for method in methods:
            last_start = _fmt_dt(diag.when_method_last_started(name, method))
            last_end   = _fmt_dt(diag.when_method_last_ended(name, method))
            running    = diag.method_currently_running(name, method)
            params     = diag.get_method_timer_parameters(name, method)
            freq       = params.frequency_minutes
            mx         = params.max_executions
            completion = " (on-completion)" if params.run_on_completion_only else ""
            print(
                f"    {method:<45s}  "
                f"last_start={last_start}  last_end={last_end}  "
                f"running={running}  freq={freq}m  max={mx}{completion}"
            )


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List production process status for the active PYSYS_UNIVERSE."
    )
    parser.add_argument(
        "--methods",
        action="store_true",
        default=False,
        help="Also show per-method last-run times and config.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report_processes(show_methods=args.methods)
    return 0


if __name__ == "__main__":
    sys.exit(main())
