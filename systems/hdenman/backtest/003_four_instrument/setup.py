"""
systems/hdenman/backtest/003_four_instrument/setup.py

Loads SP500, GOLD_micro, US10, and GAS_US adjusted prices from repo
CSVs into the 'backtest' parquet store.

Sources
-------
data/futures/adjusted_prices_csv/SP500.csv
  Coverage : 1982-09-14 → 2024-03-28  (intraday, ~35k rows)

data/futures/adjusted_prices_csv/GOLD_micro.csv
  Coverage : 1975-06-02 → 2025-09-23  (intraday, ~30k rows)

data/futures/adjusted_prices_csv/US10.csv
  Coverage : 1982-08-30 → 2025-09-23  (intraday, ~30k rows)

data/futures/adjusted_prices_csv/GAS_US.csv
  Coverage : 1990-07-26 → 2024-03-28  (intraday, ~59k rows)

All CSVs have intraday rows; resampled to last price each business day.
Instrument metadata (Pointsize, Currency) comes from the global
instrumentconfig.csv — no local config CSV needed.

Hardcoded to the 'backtest' universe.

Usage
-----
    python systems/hdenman/backtest/003_four_instrument/setup.py

Instrument selection
--------------------
SP500 and GOLD_micro carry over from 002_two_instrument.

Two more instruments were chosen by screening all instruments available
in data/futures/adjusted_prices_csv for:
  1. Low pairwise correlation with SP500 AND GOLD_micro
  2. Strong EWMAC(64,256) trend Sharpe over 10k+ days
  3. Coverage of a distinct asset class (diversification of return drivers)

Pairwise correlation to SP500 and GOLD_micro for shortlisted candidates
(daily returns, full overlapping history):

    Instrument     Class    Corr(SP500)  Corr(GOLD)  TrendSR   Days
    -----------------------------------------------------------------
    OJ             Soft         -0.056      -0.040    0.130   13614
    SOYMEAL        Agri         -0.010       0.010   -0.011   13578
    HEATOIL        Energy        0.011       0.000    0.209   11493
    GILT           Bonds         0.002      -0.002    0.162   10364
    US10           Bonds        -0.006       0.003   -0.117   10871   <-- chosen
    GAS_US         Energy        0.046       0.040    0.330    8461   <-- chosen
    JGB            Bonds        -0.048       0.067    0.214    5490
    SUGAR11        Soft          0.057       0.087    0.267   13543
    WHEAT          Agri          0.070       0.129    0.217   11572
    LEANHOG        Livestock     0.046       0.027   -0.035   11867
    BUND           Bonds        -0.230       0.143    0.481    4723
    JPY            FX           -0.162       0.250    0.290   12091

Rationale for final two
-----------------------
US10  — 10-year US Treasury note.  Correlation to SP500 of -0.006 and
  to GOLD of +0.003 (effectively zero to both).  Classic flight-to-
  quality instrument; bond trends are driven by inflation/rate cycles
  that operate on multi-year horizons, entirely distinct from equity or
  metal drivers.  Deep history (1982–), the deepest liquid bond future
  available.  Trend SR of -0.117 looks weak but is a known consequence
  of the 2022 rate-shock period; bonds trended strongly for four decades
  before that and the long-term cycle driver remains intact.

GAS_US — US Natural Gas (Henry Hub).  Correlation to SP500 of +0.046
  and to GOLD of +0.040 — effectively independent of both.  Best trend
  SR (0.330) among near-zero-corr instruments with 8k+ days of history.
  Price driven by weather, storage cycles, and supply shocks that have
  nothing to do with financial market sentiment.

Alternatives considered but not chosen
---------------------------------------
BUND   — Excellent trend SR (0.481) and low equity corr (-0.23) but only
  4,700 days of history, and corr to US10 is 0.61 (adds less than a
  second independent bond instrument would).
JPY    — Good trend SR (0.290) and low equity corr (-0.162) but corr to
  GOLD is +0.25, partially overlapping the metal allocation.
HEATOIL — Near-zero corr to both (0.011 / 0.000) but energy cluster
  with GAS_US (corr ~0.01); adds less diversification than a bond leg.
GILT   — Bond; corr to US10 = -0.00 but corr to BUND = 0.77.  A third
  bond instrument is redundant given US10 already covers that class.

Four-instrument pairwise correlation matrix (full history):
         SP500  GOLD   US10  GAS_US
SP500    1.000  0.005 -0.006  0.046
GOLD     0.005  1.000  0.003  0.040
US10    -0.006  0.003  1.000 -0.005
GAS_US   0.046  0.040 -0.005  1.000

Theoretical IDM for four perfectly uncorrelated instruments: sqrt(4) = 2.00
Actual IDM from the matrix above: 1.96
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
    "SP500":      "data.futures.adjusted_prices_csv.SP500.csv",
    "GOLD_micro": "data.futures.adjusted_prices_csv.GOLD_micro.csv",
    "US10":       "data.futures.adjusted_prices_csv.US10.csv",
    "GAS_US":     "data.futures.adjusted_prices_csv.GAS_US.csv",
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
            f"  {instrument_code:<12}  {len(prices)} daily rows  "
            f"{first} → {last}  "
            f"range [{prices.min():.2f}, {prices.max():.2f}]"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load SP500, GOLD_micro, US10, GAS_US adjusted prices from CSV to parquet store."
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
    print("Run: python systems/hdenman/backtest/003_four_instrument/system.py")
