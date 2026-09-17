import argparse
import datetime

from sysdata.data_blob import dataBlob
from sysobjects.production.process_control import controlProcess
from sysproduction.data.control_process import diagControlProcess


def process_completed_since(
    process_control: controlProcess, earliest_start: datetime.datetime
) -> bool:
    if process_control.currently_running or process_control.recently_crashed:
        return False

    last_start = process_control.last_start_time
    last_end = process_control.last_end_time
    if last_start is None or last_end is None:
        return False

    return last_start >= earliest_start and last_end >= last_start


def check_process_completed_since(process_name: str, earliest_start_epoch: float):
    earliest_start = datetime.datetime.fromtimestamp(earliest_start_epoch)
    with dataBlob(log_name="check_process_completion") as data:
        process_control = diagControlProcess(data).get_control_for_process_name(
            process_name
        )

    if not process_completed_since(process_control, earliest_start):
        raise RuntimeError(
            f"Process {process_name} did not complete successfully after "
            f"{earliest_start.isoformat(timespec='seconds')}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Verify that a managed process completed during this chain run."
    )
    parser.add_argument("process_name")
    parser.add_argument("earliest_start_epoch", type=float)
    args = parser.parse_args()
    check_process_completed_since(args.process_name, args.earliest_start_epoch)


if __name__ == "__main__":
    main()
