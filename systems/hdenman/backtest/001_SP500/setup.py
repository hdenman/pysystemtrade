"""
systems/hdenman/backtest/001_SP500/setup.py

Loads SP500 adjusted prices from the repo CSV into the 'backtest' parquet store.

Source
------
data/futures/adjusted_prices_csv/SP500.csv
  Coverage : 1982-09-14 → 2024-03-28  (intraday timestamps, ~35k rows)
  Resampled: last price each business day → daily series

The global instrumentconfig.csv already contains SP500 metadata
(Pointsize=50, Currency=USD), so no local config CSV is needed.

Hardcoded to the 'backtest' universe.

Usage
-----
    python systems/hdenman/backtest/001_SP500/setup.py
"""

import os
import argparse
from datetime import date
from typing import Optional

os.environ["PYSYS_UNIVERSE"] = "backtest"

import pandas as pd
from pathlib import Path

from syscore.fileutils import resolve_path_and_filename_for_package
from syscore.universe import scoped_path
from sysdata.parquet.parquet_access import ParquetAccess
from sysdata.parquet.parquet_adjusted_prices import parquetFuturesAdjustedPricesData
from sysobjects.adjusted_prices import futuresAdjustedPrices


INSTRUMENT_CODE = "SP500"
CSV_PATH        = "data.futures.adjusted_prices_csv.SP500.csv"

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_store() -> parquetFuturesAdjustedPricesData:
    return parquetFuturesAdjustedPricesData(
        parquet_access=ParquetAccess(scoped_path("PARQUET_DATA"))
    )


def _load_csv_prices(start_date: Optional[date] = None, end_date: Optional[date] = None) -> futuresAdjustedPrices:
    csv_file = resolve_path_and_filename_for_package(CSV_PATH)
    raw = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    raw.index = pd.to_datetime(raw.index, utc=False)
    prices = raw.iloc[:, 0]

    # CSV has intraday rows — take last price each business day
    daily = prices.resample("1B").last().dropna()
    daily.index = daily.index.normalize()

    if start_date is not None:
        daily = daily[daily.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        daily = daily[daily.index <= pd.Timestamp(end_date)]

    daily.name = "price"
    return futuresAdjustedPrices(daily)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def write_parquet_prices(start_date: Optional[date] = None, end_date: Optional[date] = None) -> None:
    store  = _price_store()
    prices = _load_csv_prices(start_date=start_date, end_date=end_date)

    if store.is_code_in_data(INSTRUMENT_CODE):
        print(f"  Removing stale {INSTRUMENT_CODE} from parquet …")
        store.delete_adjusted_prices(INSTRUMENT_CODE, are_you_sure=True)

    store.add_adjusted_prices(INSTRUMENT_CODE, prices, ignore_duplication=False)
    print(
        f"  {len(prices)} daily rows  "
        f"{prices.index[0].to_pydatetime().date()} → {prices.index[-1].to_pydatetime().date()}  "
        f"range [{prices.min():.2f}, {prices.max():.2f}]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load SP500 adjusted prices from CSV to parquet store."
    )
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"universe : backtest")
    print(f"parquet  : {scoped_path('PARQUET_DATA')}")
    print()

    print(f"Loading {INSTRUMENT_CODE} from CSV …")
    write_parquet_prices(start_date=args.start, end_date=args.end)

    print()
    print("Done.")
    print("Run: python systems/hdenman/backtest/001_SP500/system.py")
