"""
Show per-contract parquet detail for a single instrument, with an optional
comparison against the barchart CSV source directory to surface import gaps.

Run as a module::

    python -m util.db_detail EUROSTX
    python -m util.db_detail US10 --parquet-store /path/to/parquet
    python -m util.db_detail SOFR --csv-path /path/to/barchart-csv-data
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from syscore.dateutils import Frequency

from syscore.constants import arg_not_supplied
from syscore.universe import scoped_path as _scoped_path
from sysdata.config.production_config import get_production_config
from sysdata.parquet.parquet_futures_per_contract_prices import (
    CONTRACT_COLLECTION,
    from_key_to_freq_and_contract,
)

_PARQUET_INDEX_COLUMN = "index"
_DISPLAY_FREQS = {Frequency.Day, Frequency.Hour, Frequency.Mixed}
_FREQ_COLS = ["Day", "Hour", "Mixed"]   # display order


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def _resolve_parquet_store(override) -> str:
    """Mirror dataBlob.parquet_root_directory: PARQUET_DATA env var wins, then config."""
    if override is not arg_not_supplied:
        return str(override)
    if os.environ.get("PARQUET_DATA"):
        return _scoped_path("PARQUET_DATA")
    return get_production_config().get_element("parquet_store")


def _index_range_from_metadata(metadata):
    mins, maxs = [], []
    for rg_idx in range(metadata.num_row_groups):
        rg = metadata.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            if col.path_in_schema != _PARQUET_INDEX_COLUMN:
                continue
            stats = col.statistics
            if stats is not None and stats.has_min_max:
                mins.append(stats.min)
                maxs.append(stats.max)
            break
    if not mins:
        return pd.NaT, pd.NaT
    return pd.Timestamp(min(mins)), pd.Timestamp(max(maxs))


def _scan_parquet_files(contracts_dir: Path, instrument: str) -> list[dict]:
    rows = []
    for path in sorted(contracts_dir.glob(f"*{instrument}#*.parquet")):
        try:
            freq, contract = from_key_to_freq_and_contract(path.stem)
        except Exception:
            continue
        if contract.instrument_code != instrument:
            continue
        if freq not in _DISPLAY_FREQS:
            continue
        try:
            meta = pq.read_metadata(str(path))
            min_ts, max_ts = _index_range_from_metadata(meta)
            n_rows = meta.num_rows
        except (OSError, pa.ArrowInvalid):
            min_ts, max_ts, n_rows = pd.NaT, pd.NaT, 0
        rows.append(
            {
                "contract": contract.date_str,
                "freq": freq.name,
                "rows": n_rows,
                "min_ts": min_ts,
                "max_ts": max_ts,
            }
        )
    return rows


def _build_detail_table(raw: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(raw)

    # One column per frequency with row counts; missing → "-"
    pivot = (
        df.pivot_table(index="contract", columns="freq", values="rows", aggfunc="sum")
        .reindex(columns=_FREQ_COLS)
    )
    for col in pivot.columns:
        pivot[col] = pivot[col].apply(lambda x: "-" if pd.isna(x) else int(x))

    # Date range per contract (across all frequencies)
    dates = df.groupby("contract").agg(
        first_price=("min_ts", "min"),
        last_price=("max_ts", "max"),
    )

    table = pivot.join(dates).reset_index().sort_values("contract")
    table["first_price"] = pd.to_datetime(table["first_price"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    table["last_price"] = pd.to_datetime(table["last_price"]).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# CSV comparison
# ---------------------------------------------------------------------------


def _csv_contracts(csv_path: str, instrument: str) -> set[str]:
    p = Path(csv_path)
    if not p.is_dir():
        return set()
    pattern = re.compile(rf"Day_{re.escape(instrument)}_(\d{{8}})\.csv")
    return {m.group(1) for f in p.iterdir() if (m := pattern.fullmatch(f.name))}


def _resolve_csv_path(override) -> str | None:
    if override is not arg_not_supplied:
        return str(override)
    return get_production_config().get_element_or_default("barchart_path", None)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def report_instrument_detail(
    instrument: str,
    parquet_store=arg_not_supplied,
    csv_path=arg_not_supplied,
) -> pd.DataFrame:
    root = _resolve_parquet_store(parquet_store)
    contracts_dir = Path(root) / CONTRACT_COLLECTION
    print(f"Contract prices for {instrument} — store={root}\n")

    raw = _scan_parquet_files(contracts_dir, instrument)
    if not raw:
        print(f"No contracts found for {instrument}.")
        return pd.DataFrame()

    table = _build_detail_table(raw)

    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
    ):
        print(table.to_string(index=False))

    # CSV coverage comparison
    resolved_csv = _resolve_csv_path(csv_path)
    if resolved_csv:
        csv_contracts = _csv_contracts(resolved_csv, instrument)
        parquet_contracts = set(table["contract"])
        only_csv = sorted(csv_contracts - parquet_contracts)
        only_parquet = sorted(parquet_contracts - csv_contracts)

        print(f"\nCSV source: {resolved_csv}")
        print(f"  contracts in parquet : {len(parquet_contracts)}")
        print(f"  contracts in CSV     : {len(csv_contracts)}")
        if only_csv:
            print(f"  in CSV, not imported : {only_csv}")
        if only_parquet:
            print(f"  in parquet, no CSV   : {only_parquet}")
        if not only_csv and not only_parquet:
            print("  parquet and CSV coverage match")

    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-contract parquet detail for a single instrument."
    )
    parser.add_argument("instrument", help="Instrument code, e.g. EUROSTX")
    parser.add_argument(
        "--parquet-store",
        dest="parquet_store",
        default=arg_not_supplied,
        metavar="PATH",
        help="Parquet store root override",
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path",
        default=arg_not_supplied,
        metavar="PATH",
        help="Barchart CSV directory for coverage comparison "
        "(default: barchart_path from config)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report_instrument_detail(
        instrument=args.instrument,
        parquet_store=args.parquet_store,
        csv_path=args.csv_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
