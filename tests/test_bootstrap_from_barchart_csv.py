import importlib

bootstrap_module = importlib.import_module("util.bootstrap_from_barchart_csv")


class _Rows:
    def __init__(self, label):
        self.label = label
        self.index = [f"{label}-start", f"{label}-end"]

    def __len__(self):
        return 2


class _DataBlob:
    def __init__(self, log_name):
        self.log_name = log_name

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        assert len(exc_info) == 3
        return False


def test_bootstrap_recreates_roll_calendar_without_confirmation(monkeypatch):
    calls = []

    def init_prices(instrument_code, datapath, csv_config, ignore_duplication):
        assert csv_config is bootstrap_module.BARCHART_CONFIG
        calls.append(("init_prices", instrument_code, datapath, ignore_duplication))

    def assert_csv_files_present(instrument_code, datapath):
        calls.append(("assert_csv_files_present", instrument_code, datapath))

    def build_roll_calendar(instrument_code, write, check_before_writing):
        calls.append(
            ("build_roll_calendar", instrument_code, write, check_before_writing)
        )
        return _Rows("roll")

    def process_multiple_prices(instrument_code, ADD_TO_DB, ADD_TO_CSV):
        calls.append(
            ("process_multiple_prices", instrument_code, ADD_TO_DB, ADD_TO_CSV)
        )
        return _Rows("multiple")

    def process_adjusted_prices(
        instrument_code, multiple_prices, ADD_TO_DB, ADD_TO_CSV
    ):
        calls.append(
            (
                "process_adjusted_prices",
                instrument_code,
                multiple_prices.label,
                ADD_TO_DB,
                ADD_TO_CSV,
            )
        )
        return _Rows("adjusted")

    def update_active_contracts(data, instrument_code):
        calls.append(("update_active_contracts", data.log_name, instrument_code))

    monkeypatch.setattr(
        bootstrap_module,
        "_assert_barchart_csv_files_present",
        assert_csv_files_present,
    )
    monkeypatch.setattr(
        bootstrap_module,
        "init_db_with_split_freq_csv_prices_for_code",
        init_prices,
    )
    monkeypatch.setattr(
        bootstrap_module,
        "build_and_write_roll_calendar",
        build_roll_calendar,
    )
    monkeypatch.setattr(
        bootstrap_module,
        "process_multiple_prices_single_instrument",
        process_multiple_prices,
    )
    monkeypatch.setattr(
        bootstrap_module,
        "process_adjusted_prices_single_instrument",
        process_adjusted_prices,
    )
    monkeypatch.setattr(bootstrap_module, "dataBlob", _DataBlob)
    monkeypatch.setattr(
        bootstrap_module,
        "update_active_contracts_with_data",
        update_active_contracts,
    )

    bootstrap_module.bootstrap_from_barchart_csv("BUND", datapath="csv-path")

    assert calls == [
        ("assert_csv_files_present", "BUND", "csv-path"),
        ("init_prices", "BUND", "csv-path", True),
        ("build_roll_calendar", "BUND", True, False),
        ("process_multiple_prices", "BUND", True, False),
        ("process_adjusted_prices", "BUND", "multiple", True, False),
        ("update_active_contracts", "Bootstrap-Barchart-CSV", "BUND"),
    ]


def test_missing_barchart_csv_files_error_is_distinct(tmp_path):
    try:
        bootstrap_module._assert_barchart_csv_files_present("BUND", str(tmp_path))
    except bootstrap_module.MissingBarchartCsvFilesError as error:
        message = str(error)
    else:
        raise AssertionError("Expected MissingBarchartCsvFilesError")

    assert "No Barchart CSV files found for BUND" in message
    assert str(tmp_path) in message
    assert "Day_BUND_<YYYYMM00>.csv" in message
    assert "Hour_BUND_<YYYYMM00>.csv" in message
