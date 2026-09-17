import datetime

from sysobjects.production.process_control import controlProcess
from sysproduction.check_process_completion import process_completed_since


def test_process_completed_since_requires_current_successful_run():
    earliest_start = datetime.datetime(2026, 9, 17, 20, 0)
    completed = controlProcess(
        last_start_time=earliest_start + datetime.timedelta(seconds=1),
        last_end_time=earliest_start + datetime.timedelta(minutes=2),
    )

    assert process_completed_since(completed, earliest_start)


def test_process_completed_since_rejects_old_running_and_crashed_processes():
    earliest_start = datetime.datetime(2026, 9, 17, 20, 0)
    previous_day = earliest_start - datetime.timedelta(days=1)

    old = controlProcess(last_start_time=previous_day, last_end_time=previous_day)
    running = controlProcess(
        last_start_time=earliest_start,
        last_end_time=earliest_start,
        currently_running=True,
    )
    crashed = controlProcess(
        last_start_time=earliest_start,
        last_end_time=earliest_start + datetime.timedelta(minutes=1),
        recently_crashed=True,
    )

    assert not process_completed_since(old, earliest_start)
    assert not process_completed_since(running, earliest_start)
    assert not process_completed_since(crashed, earliest_start)
