"""
systems/hdenman/synthetic/003_noisy_sine/setup.py

Idempotent setup for the SYN_NOISY_SINE synthetic instrument.

Price model
-----------
    price[t] = MEAN
             + AMPLITUDE * sin(2π * calendar_days_from_start / PERIOD_DAYS)
             + N(0, NOISE_STD)

Parameters
----------
PERIOD_DAYS  Period of the sine wave in calendar days (7 = one cycle per week)
MEAN         Price midpoint
AMPLITUDE    Peak deviation of the sine component  (range: MEAN ± AMPLITUDE)
NOISE_STD    Std dev of the i.i.d. Gaussian noise layer
RANDOM_SEED  Fixes the noise draw for reproducibility

Hardcoded to the 'synthetic' universe.

Usage
-----
    python systems/hdenman/synthetic/003_noisy_sine/setup.py
"""

import os

os.environ["PYSYS_UNIVERSE"] = "synthetic"

import numpy as np
import pandas as pd
from pathlib import Path

from syscore.universe import scoped_path
from sysdata.parquet.parquet_access import ParquetAccess
from sysdata.parquet.parquet_adjusted_prices import parquetFuturesAdjustedPricesData
from sysobjects.adjusted_prices import futuresAdjustedPrices

# ---------------------------------------------------------------------------
# Parameters — edit these to change the instrument
# ---------------------------------------------------------------------------

INSTRUMENT_CODE = "SYN_NOISY_SINE"

PERIOD_DAYS  = 7*4*1      # calendar-day period (7 = one full cycle per calendar week)
MEAN         = 100.0
AMPLITUDE    = 20.0   # sine component: price swings MEAN ± AMPLITUDE

NOISE_STD    = 5.0    # std dev of i.i.d. Gaussian noise added each day
RANDOM_SEED  = 42     # fix for reproducibility

START_DATE   = pd.Timestamp("2020-01-01")

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_store() -> parquetFuturesAdjustedPricesData:
    return parquetFuturesAdjustedPricesData(
        parquet_access=ParquetAccess(scoped_path("PARQUET_DATA"))
    )


def _build_prices() -> futuresAdjustedPrices:
    end      = pd.Timestamp.today().normalize()
    index    = pd.bdate_range(start=START_DATE, end=end)
    cal_days = np.array([(dt - START_DATE).days for dt in index], dtype=float)

    sine  = AMPLITUDE * np.sin(2 * np.pi * cal_days / PERIOD_DAYS)
    noise = np.random.default_rng(RANDOM_SEED).normal(0.0, NOISE_STD, len(index))

    prices = MEAN + sine + noise
    return futuresAdjustedPrices(pd.Series(prices, index=index, dtype=float))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def write_instrument_config() -> None:
    dest = HERE / "instrumentconfig.csv"
    df = pd.DataFrame(
        [{
            "Instrument":  INSTRUMENT_CODE,
            "Description": (
                f"Synthetic sine+noise instrument "
                f"(period={PERIOD_DAYS}d, mean={MEAN}, "
                f"amplitude=±{AMPLITUDE}, noise_std={NOISE_STD})"
            ),
            "Pointsize":   1.0,
            "Currency":    "USD",
            "AssetClass":  "Synthetic",
            "PerBlock":    0.0,
            "Percentage":  0.0,
            "PerTrade":    0.0,
            "Region":      "SYNTHETIC",
        }]
    ).set_index("Instrument")
    df.to_csv(dest)
    print(f"  {dest}")


def write_parquet_prices() -> None:
    store = _price_store()
    if store.is_code_in_data(INSTRUMENT_CODE):
        print(f"  Removing stale {INSTRUMENT_CODE} from parquet …")
        store.delete_adjusted_prices(INSTRUMENT_CODE, are_you_sure=True)
    prices = _build_prices()
    store.add_adjusted_prices(INSTRUMENT_CODE, prices, ignore_duplication=False)
    print(
        f"  {len(prices)} rows  "
        f"{prices.index[0].date()} → {prices.index[-1].date()}  "
        f"range [{prices.min():.2f}, {prices.max():.2f}]"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"universe   : synthetic")
    print(f"parquet    : {scoped_path('PARQUET_DATA')}")
    print(f"wave       : MEAN={MEAN}  AMPLITUDE=±{AMPLITUDE}  PERIOD={PERIOD_DAYS} calendar days")
    print(f"noise      : N(0, {NOISE_STD})  seed={RANDOM_SEED}")
    print()

    print("[1/2] instrument config →")
    write_instrument_config()

    print()
    print("[2/2] adjusted prices → parquet …")
    write_parquet_prices()

    print()
    print("Done.")
