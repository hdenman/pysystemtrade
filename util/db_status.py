"""
Report the contents of the configured data stores.

Read-only utility that summarises:

* the live MongoDB database (collections, document counts, storage sizes), and
* the contract price parquet store (per-instrument contract counts, price
  ranges, total observations).

Connection precedence (arguments > ``private_config.yaml`` > ``defaults.yaml``)
is delegated to :class:`sysdata.mongodb.mongo_connection.mongoDb` for Mongo and
to ``get_production_config().get_element("parquet_store")`` for the parquet
root, mirroring :class:`sysdata.data_blob.dataBlob`.

Run as a module::

    python -m util.db_status
    python -m util.db_status --sample 1
    python -m util.db_status --db production --host 127.0.0.1 --port 27017
    python -m util.db_status --parquet-store /tmp/parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pymongo.errors import PyMongoError

from syscore.constants import arg_not_supplied
from syscore.exceptions import missingData
from sysdata.config.production_config import get_production_config
from sysdata.mongodb.mongo_connection import clean_mongo_host, mongoDb
from sysdata.parquet.parquet_futures_per_contract_prices import (
    CONTRACT_COLLECTION,
    from_key_to_freq_and_contract,
)
from sysdata.parquet.parquet_multiple_prices import MULTIPLE_COLLECTION
from sysdata.parquet.parquet_spotfx_prices import SPOTFX_COLLECTION

_KB = 1024
_TOTALS_LABEL = "TOTAL"
_MULTIPLE_PRICE_COLUMNS = [
    "instrument",
    "first_price",
    "last_price",
    "prices",
]
_FX_PRICE_COLUMNS = [
    "currency",
    "first_price",
    "last_price",
    "prices",
]
_PARQUET_INDEX_COLUMN = "index"
_MONGO_COLUMNS = [
    "collection",
    "documents",
    "size_kb",
    "storage_kb",
    "indexes",
    "index_size_kb",
]
_PRICE_COLUMNS = [
    "instrument",
    "contracts",
    "first_price",
    "last_price",
    "prices",
]


def report_db_status(
    mongo_db: str = arg_not_supplied,
    mongo_host: str = arg_not_supplied,
    mongo_port: int = arg_not_supplied,
    sample: int = 0,
) -> pd.DataFrame:
    """Print and return per-collection stats for the configured Mongo database.

    :param mongo_db: database name override
    :param mongo_host: hostname override
    :param mongo_port: port override
    :param sample: max sample docs to print per collection (0 disables)
    :return: DataFrame of per-collection stats with an appended TOTAL row
    :raises pymongo.errors.PyMongoError: if the server is unreachable
    """
    handle = mongoDb(
        mongo_db=mongo_db, mongo_host=mongo_host, mongo_port=mongo_port
    )

    # Force a real round-trip; pymongo otherwise lazy-connects and
    # silently defers any unreachable-host error to the first query.
    handle.client.server_info()

    names = sorted(handle.db.list_collection_names())
    table = _build_table(handle.db, names)

    _print_mongo_header(handle, table)
    _print_table(table, empty_message="(no collections)")
    if sample > 0:
        _print_samples(handle.db, names, sample)

    return table


def _build_table(db, names: list[str]) -> pd.DataFrame:
    rows = [_collection_row(db, name) for name in names]
    table = pd.DataFrame(rows, columns=_MONGO_COLUMNS)
    table = table.sort_values("documents", ascending=False, na_position="last")
    totals = {
        "collection": _TOTALS_LABEL,
        "documents": table["documents"].sum(),
        "size_kb": table["size_kb"].sum(),
        "storage_kb": table["storage_kb"].sum(),
        "indexes": table["indexes"].sum(),
        "index_size_kb": table["index_size_kb"].sum(),
    }
    return pd.concat(
        [table, pd.DataFrame([totals], columns=_MONGO_COLUMNS)],
        ignore_index=True,
    )


def _collection_row(db, name: str) -> dict:
    collection = db[name]
    try:
        documents = collection.estimated_document_count()
    except PyMongoError:
        documents = float("nan")
    try:
        # Subtract the implicit ``_id_`` index; clamp in case it's absent
        # (e.g. on a view).
        indexes = max(len(collection.index_information()) - 1, 0)
    except PyMongoError:
        indexes = float("nan")
    try:
        stats = db.command("collStats", name)
        size_kb = stats.get("size", 0) / _KB
        storage_kb = stats.get("storageSize", 0) / _KB
        index_size_kb = stats.get("totalIndexSize", 0) / _KB
    except PyMongoError:
        size_kb = storage_kb = index_size_kb = float("nan")
    return {
        "collection": name,
        "documents": documents,
        "size_kb": size_kb,
        "storage_kb": storage_kb,
        "indexes": indexes,
        "index_size_kb": index_size_kb,
    }


def _print_mongo_header(handle: mongoDb, table: pd.DataFrame) -> None:
    body = table[table["collection"] != _TOTALS_LABEL]
    print(
        "MongoDB status — host=%s, db=%s"
        % (clean_mongo_host(handle.host), handle.database_name)
    )
    print(
        "collections=%d, total documents=%d, total size=%.1f KB"
        % (len(body), int(body["documents"].sum()), body["size_kb"].sum())
    )


def _print_table(table: pd.DataFrame, *, empty_message: str) -> None:
    # Only the appended TOTAL row → no real rows worth showing.
    if len(table) <= 1:
        print(empty_message)
        return
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", None,
        "display.float_format", lambda x: f"{x:,.1f}",
    ):
        print(table.to_string(index=False))


def _print_samples(db, names: list[str], sample: int) -> None:
    for name in names:
        print()
        print(f"-- sample: {name} (limit {sample}) --")
        for doc in db[name].find().limit(sample):
            print(json.dumps(doc, default=str, indent=2, sort_keys=True))


def report_contract_prices_status(
    parquet_store: str = arg_not_supplied,
) -> pd.DataFrame:
    """Print and return per-instrument contract-price stats from the parquet store.

    Each row of the returned DataFrame represents one instrument. Multiple
    frequencies (``Day@``, ``Hour@``, mixed) for the same ``(instrument, expiry)``
    pair count as a single contract. A TOTAL row is appended.

    Per-file row counts and min/max timestamps come from the parquet footer
    (``num_rows`` + ``index`` column statistics) — no price data is read.

    :param parquet_store: parquet root override (default: ``parquet_store`` config)
    :return: DataFrame with columns ``instrument, contracts, first_price,
        last_price, prices``, plus an appended TOTAL row
    :raises syscore.exceptions.missingData: if no parquet store is configured
    """
    root = _resolve_parquet_store(parquet_store)
    contracts_dir = Path(root) / CONTRACT_COLLECTION
    print(f"Contract prices — store={root}")

    if not contracts_dir.is_dir():
        print(f"(missing {CONTRACT_COLLECTION}/ subdirectory)")
        return _empty_prices_table()

    summaries = [
        _parquet_file_summary(path)
        for path in sorted(contracts_dir.glob("*.parquet"))
    ]
    table = _aggregate_prices_by_instrument(summaries)
    _print_prices_summary(table)
    _print_table(table, empty_message="(no contract price files)")
    return table


def _resolve_parquet_store(parquet_store) -> str:
    if parquet_store is not arg_not_supplied:
        return str(parquet_store)
    return get_production_config().get_element("parquet_store")


def _parquet_file_summary(path: Path) -> dict:
    """Extract (instrument, expiry, row count, min/max ts) from a parquet footer."""
    instrument, date_str = _parse_contract_ident(path.stem)
    try:
        metadata = pq.read_metadata(str(path))
    except (OSError, pa.ArrowInvalid):
        return {
            "instrument": instrument,
            "date_str": date_str,
            "rows": 0,
            "min_ts": pd.NaT,
            "max_ts": pd.NaT,
        }
    min_ts, max_ts = _index_range_from_metadata(metadata)
    return {
        "instrument": instrument,
        "date_str": date_str,
        "rows": metadata.num_rows,
        "min_ts": min_ts,
        "max_ts": max_ts,
    }


def _parse_contract_ident(stem: str) -> tuple[str, str]:
    """``[FREQ@]INSTRUMENT#DATE`` → ``(INSTRUMENT, DATE)``.

    Delegates to the canonical parser in :mod:`sysdata.parquet.parquet_futures_per_contract_prices`
    so any future change to the on-disk key format is honoured automatically.
    """
    _, contract = from_key_to_freq_and_contract(stem)
    return contract.instrument_code, contract.date_str


def _index_range_from_metadata(metadata) -> tuple:
    """Min/max of the ``index`` (timestamp) column from parquet row-group stats."""
    mins: list = []
    maxs: list = []
    for rg_idx in range(metadata.num_row_groups):
        rg = metadata.row_group(rg_idx)
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            if col.path_in_schema != _PARQUET_INDEX_COLUMN:
                continue
            stats = col.statistics
            if stats is None or not stats.has_min_max:
                continue
            mins.append(stats.min)
            maxs.append(stats.max)
            break
    if not mins:
        return pd.NaT, pd.NaT
    return pd.Timestamp(min(mins)), pd.Timestamp(max(maxs))


def _aggregate_prices_by_instrument(summaries: list[dict]) -> pd.DataFrame:
    if not summaries:
        return _empty_prices_table()
    files = pd.DataFrame(summaries)
    contracts = (
        files.drop_duplicates(["instrument", "date_str"])
        .groupby("instrument")
        .size()
        .rename("contracts")
    )
    aggregated = files.groupby("instrument").agg(
        first_price=("min_ts", "min"),
        last_price=("max_ts", "max"),
        prices=("rows", "sum"),
    )
    table = (
        aggregated.join(contracts)
        .reset_index()[_PRICE_COLUMNS]
        .sort_values("prices", ascending=False, na_position="last")
    )
    totals = {
        "instrument": _TOTALS_LABEL,
        "contracts": int(table["contracts"].sum()),
        "first_price": table["first_price"].min(),
        "last_price": table["last_price"].max(),
        "prices": int(table["prices"].sum()),
    }
    return pd.concat(
        [table, pd.DataFrame([totals], columns=_PRICE_COLUMNS)],
        ignore_index=True,
    )


def _empty_prices_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_PRICE_COLUMNS)


def _print_prices_summary(table: pd.DataFrame) -> None:
    body = table[table["instrument"] != _TOTALS_LABEL]
    print(
        "instruments=%d, contracts=%d, prices=%d"
        % (
            len(body),
            int(body["contracts"].sum()),
            int(body["prices"].sum()),
        )
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
            "Report MongoDB collections and contract-price and multiple-price "
            "parquet contents."
    )
    parser.add_argument(
        "--db",
        dest="mongo_db",
        default=arg_not_supplied,
        help="Database name override",
    )
    parser.add_argument(
        "--host",
        dest="mongo_host",
        default=arg_not_supplied,
        help="Mongo host (or full mongodb:// URI) override",
    )
    parser.add_argument(
        "--port",
        dest="mongo_port",
        type=int,
        default=arg_not_supplied,
        help="Mongo port override",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Print up to N sample documents per collection (default: 0)",
    )
    parser.add_argument(
        "--parquet-store",
        dest="parquet_store",
        default=arg_not_supplied,
        metavar="PATH",
        help="Parquet store root override",
    )
    return parser.parse_args(argv)


def report_multiple_prices_status(
    parquet_store: str = arg_not_supplied,
) -> pd.DataFrame:
    """Print and return per-instrument multiple-price stats from the parquet store.

    Each row represents one instrument. A TOTAL row is appended.
    Row counts and min/max timestamps come from the parquet footer — no data read.

    :param parquet_store: parquet root override (default: ``parquet_store`` config)
    :return: DataFrame with columns ``instrument, first_price, last_price, prices``,
        plus an appended TOTAL row
    :raises syscore.exceptions.missingData: if no parquet store is configured
    """
    root = _resolve_parquet_store(parquet_store)
    multiple_dir = Path(root) / MULTIPLE_COLLECTION
    print(f"Multiple prices — store={root}")

    if not multiple_dir.is_dir():
        print(f"(missing {MULTIPLE_COLLECTION}/ subdirectory)")
        return _empty_multiple_prices_table()

    summaries = [
        _multiple_price_file_summary(path)
        for path in sorted(multiple_dir.glob("*.parquet"))
    ]
    table = _build_multiple_prices_table(summaries)
    _print_multiple_prices_summary(table)
    _print_table(table, empty_message="(no multiple price files)")
    return table


def _multiple_price_file_summary(path: Path) -> dict:
    """Extract (instrument, row count, min/max ts) from a parquet footer."""
    instrument = path.stem
    try:
        metadata = pq.read_metadata(str(path))
    except (OSError, pa.ArrowInvalid):
        return {
            "instrument": instrument,
            "rows": 0,
            "min_ts": pd.NaT,
            "max_ts": pd.NaT,
        }
    min_ts, max_ts = _index_range_from_metadata(metadata)
    return {
        "instrument": instrument,
        "rows": metadata.num_rows,
        "min_ts": min_ts,
        "max_ts": max_ts,
    }


def _build_multiple_prices_table(summaries: list[dict]) -> pd.DataFrame:
    if not summaries:
        return _empty_multiple_prices_table()
    files = pd.DataFrame(summaries)
    table = (
        files.rename(
            columns={"min_ts": "first_price", "max_ts": "last_price", "rows": "prices"}
        )[_MULTIPLE_PRICE_COLUMNS]
        .sort_values("prices", ascending=False, na_position="last")
    )
    totals = {
        "instrument": _TOTALS_LABEL,
        "first_price": table["first_price"].min(),
        "last_price": table["last_price"].max(),
        "prices": int(table["prices"].sum()),
    }
    return pd.concat(
        [table, pd.DataFrame([totals], columns=_MULTIPLE_PRICE_COLUMNS)],
        ignore_index=True,
    )


def _empty_multiple_prices_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_MULTIPLE_PRICE_COLUMNS)


def _print_multiple_prices_summary(table: pd.DataFrame) -> None:
    body = table[table["instrument"] != _TOTALS_LABEL]
    print(
        "instruments=%d, prices=%d"
        % (len(body), int(body["prices"].sum()))
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    failures = 0
    try:
        report_db_status(
            mongo_db=args.mongo_db,
            mongo_host=args.mongo_host,
            mongo_port=args.mongo_port,
            sample=args.sample,
        )
    except PyMongoError as exc:
        print(f"MongoDB unreachable: {exc}", file=sys.stderr)
        failures += 1

    print()
    try:
        report_contract_prices_status(parquet_store=args.parquet_store)
    except missingData as exc:
        print(f"Contract prices unavailable: {exc}", file=sys.stderr)
        failures += 1

    print()
    try:
        report_multiple_prices_status(parquet_store=args.parquet_store)
    except missingData as exc:
        print(f"Multiple prices unavailable: {exc}", file=sys.stderr)
        failures += 1

    print()
    try:
        report_fx_prices_status(parquet_store=args.parquet_store)
    except missingData as exc:
        print(f"FX prices unavailable: {exc}", file=sys.stderr)
        failures += 1

    return 1 if failures else 0


def report_fx_prices_status(
    parquet_store: str = arg_not_supplied,
) -> pd.DataFrame:
    """Print and return per-currency FX price stats from the parquet store.

    Each row represents one currency pair. A TOTAL row is appended.
    Row counts and min/max timestamps come from the parquet footer — no data read.

    :param parquet_store: parquet root override (default: ``parquet_store`` config)
    :return: DataFrame with columns ``currency, first_price, last_price, prices``,
        plus an appended TOTAL row
    :raises syscore.exceptions.missingData: if no parquet store is configured
    """
    root = _resolve_parquet_store(parquet_store)
    fx_dir = Path(root) / SPOTFX_COLLECTION
    print(f"FX prices — store={root}")

    if not fx_dir.is_dir():
        print(f"(missing {SPOTFX_COLLECTION}/ subdirectory)")
        return _empty_fx_prices_table()

    summaries = [
        _fx_price_file_summary(path)
        for path in sorted(fx_dir.glob("*.parquet"))
    ]
    table = _build_fx_prices_table(summaries)
    _print_fx_prices_summary(table)
    _print_table(table, empty_message="(no FX price files)")
    return table


def _fx_price_file_summary(path: Path) -> dict:
    """Extract (currency, row count, min/max ts) from a parquet footer."""
    currency = path.stem
    try:
        metadata = pq.read_metadata(str(path))
    except (OSError, pa.ArrowInvalid):
        return {
            "currency": currency,
            "rows": 0,
            "min_ts": pd.NaT,
            "max_ts": pd.NaT,
        }
    min_ts, max_ts = _index_range_from_metadata(metadata)
    return {
        "currency": currency,
        "rows": metadata.num_rows,
        "min_ts": min_ts,
        "max_ts": max_ts,
    }


def _build_fx_prices_table(summaries: list[dict]) -> pd.DataFrame:
    if not summaries:
        return _empty_fx_prices_table()
    files = pd.DataFrame(summaries)
    table = (
        files.rename(
            columns={"min_ts": "first_price", "max_ts": "last_price", "rows": "prices"}
        )[_FX_PRICE_COLUMNS]
        .sort_values("prices", ascending=False, na_position="last")
    )
    totals = {
        "currency": _TOTALS_LABEL,
        "first_price": table["first_price"].min(),
        "last_price": table["last_price"].max(),
        "prices": int(table["prices"].sum()),
    }
    return pd.concat(
        [table, pd.DataFrame([totals], columns=_FX_PRICE_COLUMNS)],
        ignore_index=True,
    )


def _empty_fx_prices_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_FX_PRICE_COLUMNS)


def _print_fx_prices_summary(table: pd.DataFrame) -> None:
    body = table[table["currency"] != _TOTALS_LABEL]
    print(
        "currencies=%d, prices=%d"
        % (len(body), int(body["prices"].sum()))
    )
if __name__ == "__main__":
    sys.exit(main())
