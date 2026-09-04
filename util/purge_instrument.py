"""
Purge all database records for a specified instrument from MongoDB/Arctic.

Deletes:
  - Multiple prices (arctic/db)
  - Adjusted prices (arctic/db)
  - Per-contract historical prices (arctic/db)
  - Contract metadata (mongo/db)

Usage::

    python -m util.purge_instrument EUROSTX
    python -m util.purge_instrument EUROSTX --confirm
    python -m util.purge_instrument            # interactive prompt
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from sysdata.data_blob import dataBlob
from sysproduction.data.contracts import dataContracts
from sysproduction.data.prices import updatePrices


def purge_instrument(instrument_code: str, data: Optional[dataBlob] = None) -> dict[str, str]:
    """Purge all database entries for ``instrument_code``.

    :param instrument_code: Symbol/code of the instrument (e.g. 'EUROSTX')
    :param data: Optional pre-configured dataBlob instance
    :return: Dict of data_type -> result status message
    """
    if data is None:
        data = dataBlob()

    update_prices = updatePrices(data)
    diag_contracts = dataContracts(data)

    results = {}

    # 1. Multiple prices
    try:
        if update_prices.db_futures_multiple_prices_data.is_code_in_data(instrument_code):
            update_prices.db_futures_multiple_prices_data.delete_multiple_prices(
                instrument_code, are_you_sure=True
            )
            results["multiple_prices"] = "Deleted"
        else:
            results["multiple_prices"] = "Not found (skipped)"
    except Exception as exc:
        results["multiple_prices"] = f"Error: {exc}"

    # 2. Adjusted prices
    try:
        if update_prices.db_futures_adjusted_prices_data.is_code_in_data(instrument_code):
            update_prices.db_futures_adjusted_prices_data.delete_adjusted_prices(
                instrument_code, are_you_sure=True
            )
            results["adjusted_prices"] = "Deleted"
        else:
            results["adjusted_prices"] = "Not found (skipped)"
    except Exception as exc:
        results["adjusted_prices"] = f"Error: {exc}"

    # 3. Per-contract historical prices
    try:
        update_prices.db_futures_contract_price_data.delete_merged_prices_for_instrument_code(
            instrument_code, areyousure=True
        )
        results["contract_prices"] = "Deleted"
    except Exception as exc:
        results["contract_prices"] = f"Error: {exc}"

    # 4. Contract metadata
    try:
        diag_contracts.delete_all_contracts_for_instrument(
            instrument_code, are_you_sure=True
        )
        results["contract_metadata"] = "Deleted"
    except Exception as exc:
        results["contract_metadata"] = f"Error: {exc}"

    return results


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Purge all database entries for a given instrument from MongoDB/Arctic."
    )
    parser.add_argument(
        "instrument",
        nargs="?",
        default=None,
        help="Instrument code to purge (e.g. EUROSTX). Prompted interactively if omitted.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Skip interactive confirmation prompt.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    instrument_code = args.instrument
    if not instrument_code:
        instrument_code = input("Enter instrument code to purge: ").strip()

    if not instrument_code:
        print("No instrument code specified. Aborting.", file=sys.stderr)
        return 1

    instrument_code = instrument_code.upper()

    if not args.confirm:
        confirm = input(
            f"WARNING: This will permanently delete ALL database data (prices and contract metadata) "
            f"for instrument '{instrument_code}' from MongoDB/Arctic. Continue? [y/N]: "
        )
        if confirm.lower() not in ("y", "yes"):
            print("Aborted.")
            return 0

    print(f"Purging instrument '{instrument_code}'...")
    results = purge_instrument(instrument_code)

    for data_type, status in results.items():
        print(f"  [{data_type}] {status}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
