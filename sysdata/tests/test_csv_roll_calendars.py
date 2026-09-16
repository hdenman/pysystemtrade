import yaml
import pytest
from sysdata.csv.csv_roll_calendars import (
    csvRollCalendarData,
    CSV_ROLL_CALENDAR_DIRECTORY,
)
from sysdata.config.private_config import PRIVATE_CONFIG_DIR_ENV_VAR
from sysdata.config.configdata import Config


def test_csv_roll_calendar_data_explicit_datapath(tmp_path):
    custom_path = str(tmp_path / "custom_calendars")
    data = csvRollCalendarData(datapath=custom_path)
    assert data.datapath == custom_path


def test_csv_roll_calendar_data_configured_datapath(tmp_path, monkeypatch):
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    config_file = private_dir / "private_config.yaml"

    target_store = str(tmp_path / "shared-roll-calendars")
    config_data = {"roll_calendar_store": target_store}
    config_file.write_text(yaml.dump(config_data))

    monkeypatch.setenv(PRIVATE_CONFIG_DIR_ENV_VAR, str(private_dir))
    Config.reset()

    try:
        data = csvRollCalendarData()
        assert data.datapath == target_store
    finally:
        Config.reset()


def test_csv_roll_calendar_data_fallback_default(monkeypatch):
    monkeypatch.delenv(PRIVATE_CONFIG_DIR_ENV_VAR, raising=False)
    Config.reset()

    try:
        data = csvRollCalendarData()
        assert data.datapath == CSV_ROLL_CALENDAR_DIRECTORY
    finally:
        Config.reset()
