"""
systems/hdenman/backtest/004_four_instrument_mini/setup.py

Loads SP500_micro, GOLD_micro, US10, and GAS_US_mini adjusted prices
from repo CSVs into the 'backtest' parquet store.

This is the smaller-contract equivalent of 003_four_instrument, using
micro/mini-sized contracts throughout to reduce minimum capital
requirements.

Sources
-------
data/futures/adjusted_prices_csv/SP500_micro.csv
  Coverage : 1997-12-15 → 2025-09-23  (intraday, ~25k rows)
  Pointsize: $5 × index  (vs $50 for full SP500)

data/futures/adjusted_prices_csv/GOLD_micro.csv
  Coverage : 1975-06-02 → 2025-09-23  (intraday, ~30k rows)
  Pointsize: 10 troy oz  (vs 100 for full GOLD)

data/futures/adjusted_prices_csv/US10.csv
  Coverage : 1982-08-30 → 2025-09-23  (intraday, ~30k rows)
  Pointsize: $1000  (no smaller Treasury contract with adequate history;
  US10Y_small exists in instrumentconfig but has no price data)

data/futures/adjusted_prices_csv/GAS_US_mini.csv
  Coverage : 1990-05-22 → 2025-09-23  (intraday, ~28k rows)
  Pointsize: 2500 MMBtu  (vs 10000 for full GAS_US)

All CSVs have intraday rows; resampled to last price each business day.
Instrument metadata (Pointsize, Currency) comes from the global
instrumentconfig.csv — no local config CSV needed.

Hardcoded to the 'backtest' universe.

Usage
-----
    python systems/hdenman/backtest/004_four_instrument_mini/setup.py

Instrument selection
--------------------
Same four asset classes as 003_four_instrument (Equity / Metal / Bond /
Energy) using smaller-sized contracts where available:

    003 instrument  →  004 instrument   Ratio
    SP500           →  SP500_micro       1/10  ($50 → $5 pointsize)
    GOLD_micro      →  GOLD_micro        same  (already the micro)
    US10            →  US10              same  (no smaller contract with history)
    GAS_US          →  GAS_US_mini       1/4   (10000 → 2500 MMBtu)

SP500_micro and GAS_US_mini are cloned from their large equivalents via
sysinit/futures/clone_large_to_small_contracts.py and track their parent
contracts with correlation > 0.89 over the overlapping period.

Pairwise correlation matrix for this exact instrument set
(daily returns, full overlapping history per pair):

         SP500_micro  GOLD_micro    US10  GAS_US_mini
SP500_micro    1.000       0.028  -0.208        0.056
GOLD_micro     0.028       1.000   0.003        0.039
US10          -0.208       0.003   1.000       -0.006
GAS_US_mini    0.056       0.039  -0.006        1.000

Note: SP500_micro shows a stronger negative correlation to US10 (-0.208)
than SP500 does (-0.006 in 003).  This reflects the shorter overlapping
window (from 1997, capturing the post-2000 era where the equity/bond
flight-to-quality relationship strengthened) rather than a fundamental
difference between the contracts.  The higher |corr| increases
diversification benefit: IDM for this portfolio is 2.045 vs 1.960 for 003.

Theoretical IDM for four perfectly uncorrelated instruments: sqrt(4) = 2.00
Actual IDM from the matrix above: 2.045
"""

import argparse
import os

os.environ["PYSYS_UNIVERSE"] = "backtest"

from datetime import date
from typing import Optional

import pandas as pd

from syscore.fileutils import resolve_path_and_filename_for_package
from sysdata.parquet.parquet_access import ParquetAccess
from sysdata.parquet.parquet_adjusted_prices import parquetFuturesAdjustedPricesData
from syscore.universe import scoped_path
from sysobjects.adjusted_prices import futuresAdjustedPrices


INSTRUMENTS = {
    "SP500_micro": "data.futures.adjusted_prices_csv.SP500_micro.csv",
    "GOLD_micro":  "data.futures.adjusted_prices_csv.GOLD_micro.csv",
    "US10":        "data.futures.adjusted_prices_csv.US10.csv",
    "GAS_US_mini": "data.futures.adjusted_prices_csv.GAS_US_mini.csv",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _price_store() -> parquetFuturesAdjustedPricesData:
    return parquetFuturesAdjustedPricesData(
        parquet_access=ParquetAccess(scoped_path("PARQUET_DATA"))
    )


def _load_csv_prices(
    csv_path: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> futuresAdjustedPrices:
    csv_file = resolve_path_and_filename_for_package(csv_path)
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


def write_parquet_prices(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> None:
    store = _price_store()

    for instrument_code, csv_path in INSTRUMENTS.items():
        prices = _load_csv_prices(
            csv_path, start_date=start_date, end_date=end_date
        )

        if store.is_code_in_data(instrument_code):
            print(f"  Removing stale {instrument_code} from parquet …")
            store.delete_adjusted_prices(instrument_code, are_you_sure=True)

        store.add_adjusted_prices(instrument_code, prices, ignore_duplication=False)
        first = str(prices.index[0])[:10]
        last  = str(prices.index[-1])[:10]
        print(
            f"  {instrument_code:<14}  {len(prices)} daily rows  "
            f"{first} → {last}  "
            f"range [{prices.min():.2f}, {prices.max():.2f}]"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load SP500_micro, GOLD_micro, US10, GAS_US_mini adjusted prices from CSV to parquet store."
    )
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   type=date.fromisoformat, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"universe : backtest")
    print(f"parquet  : {scoped_path('PARQUET_DATA')}")
    print()

    write_parquet_prices(start_date=args.start, end_date=args.end)

    print()
    print("Done.")
    print("Run: python systems/hdenman/backtest/004_four_instrument_mini/system.py")
