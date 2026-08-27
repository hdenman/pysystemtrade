"""
Clear stale process control records for the active universe.

After a crash or interrupted run, processes can be stuck in "running" state in
MongoDB even though no PID is actually alive.  This blocks the next scheduled
run (pysystemtrade won't start a process it thinks is already running).

Two modes:

  --auto   Mark every process whose PID is no longer alive as finished.
           Safe to run at any time — this is what the stack handler's startup
           check does internally.

  --all    Mark ALL processes as finished, regardless of PID.  Use when --auto
           misses a case (e.g. PID was reused by another process).

Run as a module::

    PYSYS_UNIVERSE=futures python -m util.clear_processes           # --auto by default
    PYSYS_UNIVERSE=futures python -m util.clear_processes --all
    PYSYS_UNIVERSE=futures python -m util.clear_processes --process run_stack_handler
"""

import argparse
import sys

from sysdata.data_blob import dataBlob
from sysproduction.data.control_process import dataControlProcess, diagControlProcess
from sysobjects.production.process_control import processNotRunning
from syscore.universe import get_universe


def _finish_if_not_running(ctrl: dataControlProcess, name: str) -> str:
    """Try to mark a single process finished; return a status string."""
    cp = ctrl.db_control_process_data.get_control_for_process_name(name)
    if not cp.currently_running:
        return "already-stopped"
    if cp.check_if_pid_running():
        return "pid-alive-skipped"
    try:
        ctrl.finish_process(name)
    except processNotRunning:
        return "already-stopped"
    return "cleared"


def clear_dead_processes(specific: str | None = None) -> dict[str, str]:
    """Clear processes whose PIDs are no longer running.

    :param specific: if given, only clear that process name
    :return: dict of process_name → outcome string
    """
    results = {}
    with dataBlob(log_name="clear_processes") as blob:
        ctrl = dataControlProcess(blob)
        diag = diagControlProcess(blob)
        names = [specific] if specific else sorted(diag.get_list_of_process_names())
        for name in names:
            results[name] = _finish_if_not_running(ctrl, name)
    return results


def clear_all_processes(specific: str | None = None) -> dict[str, str]:
    """Force-finish all (or one) process records regardless of PID.

    :param specific: if given, only clear that process name
    :return: dict of process_name → outcome string
    """
    results = {}
    with dataBlob(log_name="clear_processes") as blob:
        ctrl = dataControlProcess(blob)
        diag = diagControlProcess(blob)
        names = [specific] if specific else sorted(diag.get_list_of_process_names())
        for name in names:
            cp = ctrl.db_control_process_data.get_control_for_process_name(name)
            if not cp.currently_running:
                results[name] = "already-stopped"
                continue
            try:
                ctrl.finish_process(name)
                results[name] = "cleared"
            except processNotRunning:
                results[name] = "already-stopped"
    return results


def _print_results(results: dict[str, str]) -> None:
    for name, outcome in results.items():
        marker = "✓" if outcome == "cleared" else " "
        print(f"  {marker} {name:<45s}  {outcome}")
    cleared = sum(1 for v in results.values() if v == "cleared")
    print(f"\n  {cleared} process(es) cleared.")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clear stale process control records for the active PYSYS_UNIVERSE. "
            "Defaults to --auto (only clear processes whose PID is no longer alive)."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Clear only processes whose PID is no longer alive (default).",
    )
    mode.add_argument(
        "--all",
        dest="force_all",
        action="store_true",
        default=False,
        help="Force-finish ALL running processes regardless of PID.",
    )
    parser.add_argument(
        "--process",
        metavar="NAME",
        default=None,
        help="Operate on a single named process only.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    universe = get_universe().value
    print(f"\nUniverse : {universe}")

    if args.force_all:
        print("Mode     : force-clear all running processes")
        if not args.process:
            answer = input("Proceed? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 0
        results = clear_all_processes(specific=args.process)
    else:
        print("Mode     : clear processes with dead PIDs (--auto)")
        results = clear_dead_processes(specific=args.process)

    _print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
