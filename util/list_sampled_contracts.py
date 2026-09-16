"""
List contracts currently marked for sampling in MongoDB.

Traded / active system instruments are listed first, and the currently priced
contract for each instrument is marked with an asterisk (*).

Usage::

    python -m util.list_sampled_contracts
    python -m util.list_sampled_contracts --instrument EUROSTX
    python -m util.list_sampled_contracts --system hdenman.production
    python -m util.list_sampled_contracts --summary
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Set

from sysdata.config.configdata import Config
from sysdata.data_blob import dataBlob
from sysproduction.data.contracts import dataContracts
from sysproduction.data.positions import diagPositions
from sysproduction.data.prices import diagPrices


def get_sampled_contracts_by_instrument(
    data: dataBlob, instrument_code: Optional[str] = None
) -> dict[str, list[str]]:
    """Return a mapping of instrument_code -> list of sampled contract_date strings."""
    data_contracts = dataContracts(data)

    if instrument_code:
        instrument_codes = [instrument_code]
    else:
        instrument_codes = (
            data_contracts.db_contract_data.get_list_of_all_instruments_with_contracts()
        )

    sampled_map: dict[str, list[str]] = {}

    for code in sorted(instrument_codes):
        sampled = data_contracts.get_all_sampled_contracts(code)
        if sampled:
            sampled_map[code] = sorted([c.date_str for c in sampled])

    return sampled_map


def _instruments_for_system(system_path: str) -> list[str]:
    """Load instrument list from a system config YAML path or dotted name."""
    candidates = [
        system_path,
        f"{system_path}.system.yaml",
        f"{system_path}.yaml",
    ]
    if not system_path.startswith("systems/"):
        stem = system_path.replace(".", "/")
        candidates.extend([
            f"systems/{stem}.system.yaml",
            f"systems/{stem}.yaml",
            f"systems/{stem}",
        ])

    for cand in candidates:
        try:
            config = Config(cand)
            iw = config.get_element_or_default("instrument_weights", {})
            if iw:
                return list(iw.keys())
            il = config.get_element_or_default("instrument_list", [])
            if il:
                return list(il)
        except Exception:
            continue

    return []


def get_active_and_traded_instruments(
    data: dataBlob, system_path: Optional[str] = None
) -> Set[str]:
    """Return set of instrument codes that are traded or in active system configs."""
    active_instruments: set[str] = set()

    # 1. Traded instruments with positions in database
    try:
        diag_positions = diagPositions(data)
        current_pos = diag_positions.get_list_of_instruments_with_current_positions()
        active_instruments.update(current_pos)
        any_pos = diag_positions.get_list_of_instruments_with_any_position()
        active_instruments.update(any_pos)
    except Exception:
        pass

    # 2. System config instruments
    system_paths = []
    if system_path:
        system_paths.append(system_path)
    else:
        system_paths.extend([
            "systems/hdenman/production/system.yaml",
            "hdenman.production",
        ])

    for path in system_paths:
        insts = _instruments_for_system(path)
        active_instruments.update(insts)

    return active_instruments


def get_priced_contract_id(data: dataBlob, instrument_code: str) -> Optional[str]:
    """Return the currently priced contract date_str for an instrument, if available."""
    try:
        diag_prices = diagPrices(data)
        multiple_prices = diag_prices.get_multiple_prices(instrument_code)
        if not multiple_prices.empty:
            return multiple_prices.current_contract_dict().price
    except Exception:
        pass

    try:
        data_contracts = dataContracts(data)
        return data_contracts.get_priced_contract_id(instrument_code)
    except Exception:
        pass

    return None


def format_contracts_with_priced_asterisk(
    contracts: list[str], priced_contract_id: Optional[str]
) -> list[str]:
    """Append '*' to the contract date matching priced_contract_id."""
    if not priced_contract_id:
        return list(contracts)

    return [f"{c}*" if c == priced_contract_id else c for c in contracts]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List futures contracts currently marked for sampling in database. "
            "Traded / active system instruments are shown first; "
            "the currently priced contract is marked with an asterisk (*)."
        )
    )
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        default=None,
        help="Optional instrument code filter (e.g. EUROSTX).",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system YAML path or dotted name (e.g. hdenman.production).",
    )
    parser.add_argument(
        "-s",
        "--summary",
        action="store_true",
        default=False,
        help="Display summary count per instrument instead of full contract list.",
    )
    return parser.parse_args(argv)


def _print_instrument_group(
    header: str,
    group_codes: list[str],
    sampled_map: dict[str, list[str]],
    priced_map: dict[str, Optional[str]],
    summary: bool,
) -> None:
    if not group_codes:
        return

    print(f"{header}:")
    for code in group_codes:
        contracts = sampled_map[code]
        priced_id = priced_map.get(code)
        formatted_contracts = format_contracts_with_priced_asterisk(contracts, priced_id)

        if summary:
            first = contracts[0]
            last = contracts[-1]
            priced_str = f" (priced: {priced_id}*)" if priced_id else ""
            print(f"  {code:15s}: {len(contracts):2d} contract(s) [{first} -> {last}]{priced_str}")
        else:
            contracts_str = ", ".join(formatted_contracts)
            print(f"  {code:15s}: {contracts_str}")
    print()


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    with dataBlob(log_name="list_sampled_contracts") as blob:
        sampled_map = get_sampled_contracts_by_instrument(
            blob, instrument_code=args.instrument
        )

        if not sampled_map:
            if args.instrument:
                print(f"No sampled contracts found for instrument '{args.instrument}'.")
            else:
                print("No sampled contracts found in database.")
            return 0

        # Fetch priced contracts for each instrument in sampled_map
        priced_map = {
            code: get_priced_contract_id(blob, code) for code in sampled_map
        }

        # Fetch set of traded / active system instruments
        active_set = get_active_and_traded_instruments(blob, system_path=args.system)

        traded_active_codes = sorted([code for code in sampled_map if code in active_set])
        other_codes = sorted([code for code in sampled_map if code not in active_set])

        total_contracts = sum(len(contracts) for contracts in sampled_map.values())
        print(f"Sampled Contracts ({len(sampled_map)} instruments, {total_contracts} contracts total):")
        print("(* = currently priced contract)\n")

        if args.instrument or not other_codes:
            # Single group display if specific instrument filtered or all instruments are active
            codes_to_show = sorted(sampled_map.keys())
            _print_instrument_group(
                "Sampled Instruments",
                codes_to_show,
                sampled_map,
                priced_map,
                args.summary,
            )
        else:
            _print_instrument_group(
                "Traded / Active System Instruments",
                traded_active_codes,
                sampled_map,
                priced_map,
                args.summary,
            )
            _print_instrument_group(
                "Other Sampled Instruments",
                other_codes,
                sampled_map,
                priced_map,
                args.summary,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
