"""
systems/hdenman/backtest/002_two_instrument/setup.py

Loads SP500 and GOLD_micro adjusted prices from repo CSVs into the
'backtest' parquet store.

Sources
-------
data/futures/adjusted_prices_csv/SP500.csv
  Coverage : 1982-09-14 → 2024-03-28  (intraday timestamps, ~35k rows)
  Resampled: last price each business day → daily series

data/futures/adjusted_prices_csv/GOLD_micro.csv
  Coverage : 1975-06-02 → 2025-09-23  (intraday timestamps, ~30k rows)
  Resampled: last price each business day → daily series

Instrument metadata (Pointsize, Currency) comes from the global
instrumentconfig.csv — no local CSV override needed.

Hardcoded to the 'backtest' universe.

Usage
-----
    python systems/hdenman/backtest/002_two_instrument/setup.py

Instrument selection
--------------------
SP500 is the benchmark equity-index trend instrument.

GOLD_micro was chosen as its complement after screening all instruments
available in data/futures/adjusted_prices_csv for correlation to SP500
daily returns and EWMAC(64,256) trend Sharpe (180-day MRCI window for
context).  Results for low-correlation candidates (|corr| < 0.15,
n >= 1000 days):

    Instrument        Corr(SP500)   Trend SR   Days
    -----------------------------------------------
    OJ                    -0.056      0.130    10398
    IRON                  -0.023     -0.203     2589
    SOYMEAL               -0.010     -0.011    10444
    ETHANOL               -0.009      0.196     4731
    STEEL                  0.002     -0.254     3850
    GOLD                   0.005      0.263    10433
    GOLD_micro             0.005      0.354    10419   <-- chosen
    GOLD-mini              0.006      0.262    10440
    HEATOIL                0.011      0.209    10411
    OAT                    0.017      0.577     2954
    RICE                   0.018      0.123     8926
    CRUDE_W_micro          0.022     -0.244    10244
    REDWHEAT               0.024      0.099     7168
    MILKDRY                0.038      0.700     2455
    MILKWET                0.043      0.070     5277
    GAS_US                 0.046      0.330     8446
    LEANHOG                0.046     -0.035    10433
    GAS_US_mini            0.052      0.222     8477
    SUGAR16                0.052      0.652     2850
    GAS-PEN                0.055      0.242     4339
    GAS-LAST               0.057      0.402     4344
    SUGAR11                0.057      0.267    10391
    WHEAT                  0.070      0.217    10445
    SILVER                 0.076      0.044    10491
    ROBUSTA                0.089      0.182     3825
    FEEDCOW                0.089      0.052    10447
    COFFEE                 0.100      0.135     4289
    CORN                   0.101      0.057    10439
    LIVECOW                0.112      0.015    10436
    PLAT                   0.136     -0.015    10408
    SOYOIL                 0.140      0.154    10444

GOLD_micro has the best trend Sharpe (0.35) among instruments with
essentially zero long-run correlation to SP500 (+0.005).  It also has
the deepest history (10,400+ days), lowest SR cost (0.088), and a
well-behaved roll calendar.  GOLD/GOLD-mini are equivalent; the micro
contract is used because it matches the SP500_micro sizing tier.

LEANHOG and CRUDE_W_micro are near-zero correlation but trend poorly
(-0.035 and -0.244 SR respectively).  GAS_US is a strong secondary
candidate (SR 0.33) but has high vol and roll costs.
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
        description="Load SP500 and GOLD_micro adjusted prices from CSV to parquet store."
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
    print("Run: python systems/hdenman/backtest/002_two_instrument/system.py")
