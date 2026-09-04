"""
Bootstrap a futures instrument from Barchart daily CSV files.

Canonical flow used here:
  1. Import per-contract Barchart CSV prices into the production price store.
  2. Recreate the repo roll calendar CSV from the imported contract prices.
  3. Build multiple prices from contract prices and the recreated roll calendar.
  4. Build adjusted prices from multiple prices.
  5. Run update_sampled_contracts for the instrument, so contract metadata and
     sampling flags are initialised by the normal production path.

Usage::

    python -m util.bootstrap_from_barchart_csv EUROSTX
    python -m util.bootstrap_from_barchart_csv          # interactive prompt
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Optional, cast

from syscore.dateutils import DAILY_PRICE_FREQ, HOURLY_FREQ

from sysdata.config.production_config import get_production_config
from sysdata.data_blob import dataBlob
from sysdata.csv.csv_futures_contract_prices import csvFuturesContractPriceData
from sysinit.futures.adjustedprices_from_db_multiple_to_db import (
    process_adjusted_prices_single_instrument,
)
from sysinit.futures.contract_prices_from_split_freq_csv_to_db import (
    BARCHART_CONFIG,
    init_db_with_split_freq_csv_prices_for_code,
)
from sysinit.futures.rollcalendars_from_db_prices_to_csv import (
    build_and_write_roll_calendar,
)
from sysinit.futures.multipleprices_from_db_prices_and_csv_calendars_to_db import (
    process_multiple_prices_single_instrument,
)
from sysproduction.data.prices import get_valid_instrument_code_from_user
from sysproduction.update_historical_prices import (
    update_historical_prices_for_instrument,
)
from sysproduction.update_multiple_adjusted_prices import (
    update_multiple_adjusted_prices_with_data,
)
from sysproduction.update_sampled_contracts import update_active_contracts_with_data
from sysdata.tools.cleaner import get_config_for_price_filtering


class MissingBarchartCsvFilesError(Exception):
    pass


def _assert_barchart_csv_files_present(instrument_code: str, datapath: str) -> None:
    csv_prices = csvFuturesContractPriceData(
        cast(Any, datapath), config=cast(Any, BARCHART_CONFIG)
    )
    daily_contracts = (
        csv_prices.contract_dates_with_price_data_at_frequency_for_instrument_code(
            instrument_code, DAILY_PRICE_FREQ
        )
    )
    hourly_contracts = (
        csv_prices.contract_dates_with_price_data_at_frequency_for_instrument_code(
            instrument_code, HOURLY_FREQ
        )
    )

    if len(daily_contracts) == 0 and len(hourly_contracts) == 0:
        raise MissingBarchartCsvFilesError(
            f"No Barchart CSV files found for {instrument_code} in {datapath}. "
            f"Expected files named Day_{instrument_code}_<YYYYMM00>.csv or "
            f"Hour_{instrument_code}_<YYYYMM00>.csv."
        )


def _default_barchart_path() -> str:
    return get_production_config().get_element("barchart_path")


def bootstrap_from_barchart_csv(
    instrument_code: str,
    datapath: Optional[str] = None,
    update_historical: bool = False,
) -> None:
    """Bootstrap ``instrument_code`` using standard pysystemtrade data flows."""
    if datapath is None:
        datapath = _default_barchart_path()

    print(f"Bootstrapping {instrument_code} from Barchart CSV path: {datapath}")
    _assert_barchart_csv_files_present(instrument_code, datapath)

    print("\n[1/5] Importing per-contract prices from Barchart CSV")
    init_db_with_split_freq_csv_prices_for_code(
        instrument_code,
        datapath,
        csv_config=cast(Any, BARCHART_CONFIG),
        ignore_duplication=True,
    )

    print("\n[2/5] Recreating roll calendar from contract prices")
    roll_calendar = build_and_write_roll_calendar(
        instrument_code,
        write=True,
        check_before_writing=False,
    )
    print(
        f"Recreated roll calendar: {len(roll_calendar)} rows, "
        f"{roll_calendar.index[0]} -> {roll_calendar.index[-1]}"
    )

    print("\n[3/5] Building multiple prices from contract prices and roll calendar")
    multiple_prices = process_multiple_prices_single_instrument(
        instrument_code,
        ADD_TO_DB=True,
        ADD_TO_CSV=False,
    )
    print(
        f"Built multiple prices: {len(multiple_prices)} rows, "
        f"{multiple_prices.index[0]} -> {multiple_prices.index[-1]}"
    )

    print("\n[4/5] Building adjusted prices from multiple prices")
    adjusted_prices = process_adjusted_prices_single_instrument(
        instrument_code,
        multiple_prices=multiple_prices,
        ADD_TO_DB=True,
        ADD_TO_CSV=False,
    )
    print(
        f"Built adjusted prices: {len(adjusted_prices)} rows, "
        f"{adjusted_prices.index[0]} -> {adjusted_prices.index[-1]}"
    )

    print("\n[5/5] Running update_sampled_contracts path for contract metadata")
    with dataBlob(log_name="Bootstrap-Barchart-CSV") as data:
        update_active_contracts_with_data(data, instrument_code=instrument_code)

        if update_historical:
            print(
                "\n[extra] Updating sampled contract prices from IB, then refreshing "
                "multiple/adjusted prices"
            )
            cleaning_config = get_config_for_price_filtering(data)
            update_historical_prices_for_instrument(
                instrument_code,
                data,
                cleaning_config=cleaning_config,
                interactive_mode=False,
            )
            update_multiple_adjusted_prices_with_data(
                data, instrument_code=instrument_code
            )

    print(f"\nBootstrap complete for {instrument_code}.")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap one futures instrument from Barchart daily CSV files."
    )
    parser.add_argument(
        "instrument",
        nargs="?",
        default=None,
        help="Instrument code to bootstrap (e.g. EUROSTX). Prompted interactively if omitted.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        metavar="PATH",
        help="Barchart CSV directory. Defaults to barchart_path from production config.",
    )
    parser.add_argument(
        "--update-historical",
        action="store_true",
        default=False,
        help=(
            "After update_sampled_contracts, download/update sampled contract prices "
            "from IB and refresh multiple/adjusted prices."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    instrument_code = args.instrument
    if not instrument_code:
        with dataBlob(log_name="Bootstrap-Barchart-CSV") as data:
            instrument_code = get_valid_instrument_code_from_user(data=data)

    if not instrument_code:
        print("No instrument code specified. Aborting.", file=sys.stderr)
        return 1

    try:
        bootstrap_from_barchart_csv(
            instrument_code.strip(), # .upper(),
            datapath=args.csv_path,
            update_historical=args.update_historical,
        )
    except MissingBarchartCsvFilesError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
