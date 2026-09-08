import pytest
from sysdata.config.configdata import Config
from sysdata.config.private_config import get_private_config_path
from syslogdiag.emailing import get_email_details


def test_email_details_missing_keys_logs_path(caplog, monkeypatch):
    monkeypatch.setenv("PYSYS_PRIVATE_CONFIG_DIR", "/nonexistent/private/path")
    if hasattr(Config, "evaluated"):
        delattr(Config, "evaluated")

    private_config_path = get_private_config_path()
    with pytest.raises(Exception) as exc_info:
        get_email_details()

    assert private_config_path in str(exc_info.value)
    assert "/nonexistent/private/path" in private_config_path
    assert any(
        "Private config path used: " + private_config_path in record.message
        for record in caplog.records
    )
