"""
Report the contents of the configured MongoDB instance.

Read-only utility that enumerates the live collections in the configured
database and prints per-collection document counts, storage sizes, and
index counts. Connection precedence (arguments > ``private_config.yaml``
> ``defaults.yaml``) is delegated to :class:`sysdata.mongodb.mongo_connection.mongoDb`.

Run as a module::

    python -m util.db_status
    python -m util.db_status --sample 1
    python -m util.db_status --db production --host 127.0.0.1 --port 27017
"""

import argparse
import json
import sys

import pandas as pd
from pymongo.errors import PyMongoError

from syscore.constants import arg_not_supplied
from sysdata.mongodb.mongo_connection import clean_mongo_host, mongoDb

_KB = 1024
_TOTALS_LABEL = "TOTAL"
_COLUMNS = [
    "collection",
    "documents",
    "size_kb",
    "storage_kb",
    "indexes",
    "index_size_kb",
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

    _print_header(handle, table)
    _print_table(table)
    if sample > 0:
        _print_samples(handle.db, names, sample)

    return table


def _build_table(db, names: list[str]) -> pd.DataFrame:
    rows = [_collection_row(db, name) for name in names]
    table = pd.DataFrame(rows, columns=_COLUMNS)
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
        [table, pd.DataFrame([totals], columns=_COLUMNS)], ignore_index=True
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


def _print_header(handle: mongoDb, table: pd.DataFrame) -> None:
    body = table[table["collection"] != _TOTALS_LABEL]
    print(
        "MongoDB status — host=%s, db=%s"
        % (clean_mongo_host(handle.host), handle.database_name)
    )
    print(
        "collections=%d, total documents=%d, total size=%.1f KB"
        % (len(body), int(body["documents"].sum()), body["size_kb"].sum())
    )


def _print_table(table: pd.DataFrame) -> None:
    # Only the appended TOTAL row → no real collections.
    if len(table) <= 1:
        print("(no collections)")
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the contents of the configured MongoDB instance.",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report_db_status(
            mongo_db=args.mongo_db,
            mongo_host=args.mongo_host,
            mongo_port=args.mongo_port,
            sample=args.sample,
        )
    except PyMongoError as exc:
        print(f"MongoDB unreachable: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
