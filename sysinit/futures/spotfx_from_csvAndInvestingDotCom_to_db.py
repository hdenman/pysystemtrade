"""
Get FX prices from investing.com files, merge with existing CSV data (new data
wins on overlap), and write results back to CSV and/or the DB.

Investing.com CSV format assumptions:
  - UTF-8 with BOM
  - Columns: "Date", "Price" (plus others that are ignored)
  - Date format: %m/%d/%Y
  - Rows in reverse-chronological order

Filename → fx code mapping:
  "EUR_USD Historical Data.csv" → "EURUSD"   (as-is: USD per EUR)
  "GBP_USD Historical Data.csv" → "GBPUSD"   (as-is: USD per GBP)
  "USD_CNH Historical Data.csv" → "CNHUSD"   (reciprocal: file holds CNH per USD)

  USD-base detection: if the pair starts with "USD_" after stripping
  " Historical Data", prices are inverted (1/price) before merging.
"""

import argparse
import os
import logging

import pandas as pd

from sysdata.csv.csv_spot_fx import csvFxPricesData
from sysproduction.data.currency_data import dataCurrency
from sysobjects.spot_fx_prices import fxPrices

INVESTING_DOT_COM_DATE_COLUMN = "Date"
INVESTING_DOT_COM_PRICE_COLUMN = "Price"
INVESTING_DOT_COM_DATE_FORMAT = "%m/%d/%Y"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _investingdotcom_stem_to_code(stem: str) -> str:
    """
    Convert an investing.com CSV file stem to an fx code.

    "EUR_USD Historical Data" -> "EURUSD"   (USD per EUR, no inversion)
    "USD_CNH Historical Data" -> "CNHUSD"   (CNH per USD, needs inversion)
    """
    pair = stem.replace(" Historical Data", "")
    if pair.upper().startswith("USD_"):
        foreign = pair.split("_")[1]
        return (foreign + "USD").upper()
    return pair.replace("_", "").upper()


def _investingdotcom_stem_needs_invert(stem: str) -> bool:
    """Return True when the file records foreign-per-USD (must be inverted)."""
    pair = stem.replace(" Historical Data", "")
    return pair.upper().startswith("USD_")


def _read_investingdotcom_file(filepath: str, invert: bool = False) -> fxPrices:
    """
    Read an investing.com-format CSV and return a sorted fxPrices series.

    Handles UTF-8 BOM, reverse-chronological ordering, and the
    investing.com-specific Date / Price column names.

    Parameters
    ----------
    invert : bool
        When True the file records prices as foreign-per-USD; apply 1/price
        so the series is in USD-per-foreign convention.
    """
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    price_series = df[INVESTING_DOT_COM_PRICE_COLUMN].astype(float)
    price_series.index = pd.to_datetime(
        df[INVESTING_DOT_COM_DATE_COLUMN], format=INVESTING_DOT_COM_DATE_FORMAT
    )
    price_series.index.name = "index"
    if invert:
        price_series = 1.0 / price_series
    return fxPrices(price_series.sort_index())


def _discover_investingdotcom_codes(datapath: str) -> dict:
    """
    Scan *datapath* for *.csv files and return {fx_code: (filepath, invert)}.

    Only files whose name contains " Historical Data" are considered, so
    stray CSVs in the same directory are silently ignored.

    The ``invert`` flag in the tuple is True for USD-base files (e.g.
    ``USD_CNH Historical Data.csv``) where prices must be reciprocated before
    merging into the pysystemtrade USD-per-foreign convention.
    """
    result = {}
    for entry in os.scandir(datapath):
        if not entry.name.endswith(".csv"):
            continue
        stem = entry.name[:-4]  # strip ".csv"
        if "Historical Data" not in stem:
            continue
        code = _investingdotcom_stem_to_code(stem)
        result[code] = (entry.path, _investingdotcom_stem_needs_invert(stem))
    return result


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def spotfx_from_csv_and_investing_dot_com(
    datapath: str, ADD_TO_DB: bool = True, ADD_TO_CSV: bool = True
):
    """
    Merge investing.com CSV data with existing pysystemtrade CSV data, then
    write results to CSV and/or the production DB.

    Parameters
    ----------
    datapath : str
        Directory containing investing.com CSV files.
    ADD_TO_DB : bool
        Write merged prices to the production database.
    ADD_TO_CSV : bool
        Overwrite the existing pysystemtrade CSV files with merged data.
    """
    new_data_by_code = _discover_investingdotcom_codes(datapath)
    my_csv_fx_prices_data = csvFxPricesData()
    db_fx_prices_data = dataCurrency().db_fx_prices_data

    list_of_ccy_codes = my_csv_fx_prices_data.get_list_of_fxcodes()

    for currency_code in list_of_ccy_codes:
        existing = my_csv_fx_prices_data.get_fx_prices(currency_code)
        n_existing = len(existing)

        if currency_code in new_data_by_code:
            filepath, invert = new_data_by_code[currency_code]
            new = _read_investingdotcom_file(filepath, invert=invert)
            n_new = len(new)

            # new data wins on overlap: concat existing then new, keep last
            merged_series = (
                pd.concat([existing, new])
                .groupby(level=0)
                .last()
                .sort_index()
            )
            merged = fxPrices(merged_series)
            n_merged = len(merged)

            print(
                f"{currency_code}: existing={n_existing}, "
                f"file={n_new}, added={n_merged - n_existing}, total={n_merged}"
            )

            if ADD_TO_CSV:
                my_csv_fx_prices_data.add_fx_prices(
                    currency_code, merged, ignore_duplication=True
                )

            if ADD_TO_DB:
                db_fx_prices_data.add_fx_prices(
                    code=currency_code,
                    fx_price_data=merged,
                    ignore_duplication=True,
                )
        else:
            # No new investing.com file — push existing CSV to DB unchanged
            print(f"{currency_code}: existing={n_existing} rows, no new data")

            if ADD_TO_DB:
                db_fx_prices_data.add_fx_prices(
                    code=currency_code,
                    fx_price_data=existing,
                    ignore_duplication=True,
                )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge investing.com FX CSVs with pysystemtrade CSV data and write to CSV/DB."
    )
    parser.add_argument(
        "datapath",
        help="Directory containing investing.com CSV files "
             "(e.g. '~/pysystemtrade-data/investing.com-data')",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        default=False,
        help="Skip writing to the production database.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        default=False,
        help="Skip overwriting the pysystemtrade CSV files.",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("PYSYSTEMTRADE_LOG_LEVEL", "ERROR"),
        metavar="LEVEL",
        help="Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL). "
             "Also read from $PYSYSTEMTRADE_LOG_LEVEL. Default: ERROR.",
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    spotfx_from_csv_and_investing_dot_com(
        datapath=os.path.expanduser(args.datapath),
        ADD_TO_DB=not args.no_db,
        ADD_TO_CSV=not args.no_csv,
    )
