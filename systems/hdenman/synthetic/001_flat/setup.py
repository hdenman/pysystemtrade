"""
systems/hdenman/synthetic/001_flat/setup.py

Idempotent setup for the SYN_FLAT synthetic instrument.

What this creates
-----------------
Parquet  $PARQUET_DATA/synthetic/futures_adjusted_prices/SYN_FLAT.parquet
           Price 100.0 + Gaussian noise (σ=5), business-day frequency, 2020-01-01 → today

CSV      systems/hdenman/synthetic/001_flat/instrumentconfig.csv
           Pointsize=1, Currency=USD, AssetClass=Synthetic

Destructive only for SYN_FLAT: any existing parquet file is deleted and
rewritten.  No other instruments or files are touched.

Hardcoded to the 'synthetic' universe.

Usage
-----
    python systems/hdenman/synthetic/001_flat/setup.py

The PARQUET_DATA environment variable must be set (see devenv.nix).
"""

import os

# Must precede all pysystemtrade imports — universe_subdir() is called
# dynamically but scoped_path() is called at write time.
os.environ["PYSYS_UNIVERSE"] = "synthetic"

import numpy as np
import pandas as pd
from pathlib import Path

from syscore.universe import scoped_path
from sysdata.parquet.parquet_access import ParquetAccess
from sysdata.parquet.parquet_adjusted_prices import parquetFuturesAdjustedPricesData
from sysobjects.adjusted_prices import futuresAdjustedPrices

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENT_CODE = "SYN_FLAT"
PRICE = 1000.0
NOISE_STD = 5.0
RANDOM_SEED = 42          # reproducible runs
START_DATE = pd.Timestamp("2020-01-01")

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _price_store() -> parquetFuturesAdjustedPricesData:
    parquet_root = scoped_path("PARQUET_DATA")
    return parquetFuturesAdjustedPricesData(parquet_access=ParquetAccess(parquet_root))


def _build_prices() -> futuresAdjustedPrices:
    end = pd.Timestamp.today().normalize()
    index = pd.bdate_range(start=START_DATE, end=end)
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=NOISE_STD, size=len(index))
    series = pd.Series(PRICE + noise, index=index, dtype=float)
    return futuresAdjustedPrices(series)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def write_instrument_config() -> None:
    """
    Write instrumentconfig.csv to this directory.

    Column layout matches data/futures/csvconfig/instrumentconfig.csv so that
    csvFuturesInstrumentData(datapath='systems.hdenman.synthetic.001_flat')
    can read it directly.

    Costs are zero: SYN_FLAT is a frictionless synthetic instrument.
    """
    dest = HERE / "instrumentconfig.csv"
    df = pd.DataFrame(
        [
            {
                "Instrument": INSTRUMENT_CODE,
                "Description": "Synthetic flat-price spot instrument (no roll, no carry)",
                "Pointsize": 1.0,
                "Currency": "USD",
                "AssetClass": "Synthetic",
                "PerBlock": 0.0,
                "Percentage": 0.0,
                "PerTrade": 0.0,
                "Region": "SYNTHETIC",
            }
        ]
    ).set_index("Instrument")
    df.to_csv(dest)
    print(f"  {dest}")


def write_parquet_prices() -> None:
    """
    Delete any existing SYN_FLAT entry from the synthetic parquet store, then
    write a fresh constant-price series.
    """
    store = _price_store()

    if store.is_code_in_data(INSTRUMENT_CODE):
        print(f"  Removing stale {INSTRUMENT_CODE} from parquet …")
        store.delete_adjusted_prices(INSTRUMENT_CODE, are_you_sure=True)

    prices = _build_prices()
    store.add_adjusted_prices(INSTRUMENT_CODE, prices, ignore_duplication=False)

    first = prices.index[0].date()
    last = prices.index[-1].date()
    print(f"  {len(prices)} rows  {first} → {last}  price={PRICE}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parquet_root = scoped_path("PARQUET_DATA")
    print(f"universe : synthetic")
    print(f"parquet  : {parquet_root}")
    print()

    print("[1/2] instrument config →")
    write_instrument_config()

    print()
    print("[2/2] adjusted prices → parquet …")
    write_parquet_prices()

    print()
    print("Done.")
    print()
    print("How to consume:")
    print("  from sysdata.sim.db_futures_sim_data import dbFuturesSimData")
    print("  data = dbFuturesSimData(csv_data_paths={")
    print("      'csvFuturesInstrumentData': 'systems.hdenman.synthetic.001_flat',")
    print("  })")
    print(f"  prices = data.get_backadjusted_futures_price('{INSTRUMENT_CODE}')")
