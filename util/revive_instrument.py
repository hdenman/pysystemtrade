"""
Revive a futures instrument whose multiple-price state is stuck on old contracts.

This handles the failure mode where:

  * historical per-contract prices may exist for active contracts;
  * multiple / adjusted prices still point at expired PRICE/FORWARD contracts;
  * interactive_update_roll_status crashes because the old PRICE contract is not
    present in the contract metadata database.

The utility creates missing contract metadata for the roll path, rolls multiple
and adjusted prices forward until the priced contract is no longer past its
roll date, refreshes sampled contract metadata, downloads current contract
prices from IB, and finally runs the normal multiple/adjusted update.

It does not invent missing historical prices. If the instrument has a long gap
and IB no longer exposes expired contracts, old business-day gaps can remain;
the point is to make the instrument live again so daily updates can continue.

Usage::

    python -m util.revive_instrument COPPER-mini
    python -m util.revive_instrument COPPER-mini --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass
from typing import Optional

from syscore.exceptions import ContractNotFound
from sysdata.data_blob import dataBlob
from sysdata.tools.cleaner import get_config_for_price_filtering
from sysobjects.contract_dates_and_expiries import contractDate
from sysobjects.contracts import futuresContract
from sysobjects.instruments import futuresInstrument
from sysobjects.rolls import contractDateWithRollParameters
from sysproduction.data.contracts import dataContracts
from sysproduction.data.prices import diagPrices, get_valid_instrument_code_from_user
from sysproduction.reporting.data.rolls import rollingAdjustedAndMultiplePrices
from sysproduction.update_historical_prices import update_historical_prices_for_instrument
from sysproduction.update_multiple_adjusted_prices import (
    update_multiple_adjusted_prices_for_instrument,
)
from sysproduction.update_sampled_contracts import update_active_contracts_with_data


@dataclass(frozen=True)
class ReviveOptions:
    as_of_date: datetime.date
    max_rolls: int = 48
    allow_forward_fill: bool = True
    skip_sampled_contracts: bool = False
    skip_price_download: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class ReviveResult:
    starting_priced_contract: str
    ending_priced_contract: str
    rolls_performed: int
    refreshed_sampled_contracts: bool
    downloaded_prices: bool
    refreshed_multiple_adjusted: bool


def revive_instrument(instrument_code: str, options: ReviveOptions) -> ReviveResult:
    """Revive ``instrument_code`` using production data stores and normal update paths."""
    with dataBlob(log_name="Revive-Instrument") as data:
        diag_prices = diagPrices(data)

        starting_priced_contract = _current_priced_contract(diag_prices, instrument_code)
        print(
            f"Reviving {instrument_code}: starting priced contract "
            f"{starting_priced_contract}"
        )

        rolls_performed = _roll_until_priced_contract_is_live(
            data=data,
            instrument_code=instrument_code,
            options=options,
        )

        ending_priced_contract = _current_priced_contract(diag_prices, instrument_code)
        print(
            f"Roll stage complete for {instrument_code}: {rolls_performed} roll(s), "
            f"priced contract now {ending_priced_contract}"
        )

        refreshed_sampled_contracts = False
        if not options.skip_sampled_contracts:
            print("\nRefreshing sampled contract metadata")
            if options.dry_run:
                print("DRY RUN: would run update_active_contracts_with_data")
            else:
                # This is the normal production path that marks current contracts
                # as sampled and asks IB for real expiry dates. It also turns off
                # sampling for stale contracts once IB confirms they are gone.
                update_active_contracts_with_data(data, instrument_code=instrument_code)
            refreshed_sampled_contracts = True

        downloaded_prices = False
        if not options.skip_price_download:
            print("\nRefreshing historical contract prices")
            if options.dry_run:
                print("DRY RUN: would run update_historical_prices_for_instrument")
            else:
                # Pass the real cleaner config. Calling the lower-level function
                # without this object leaves the sentinel in place and crashes
                # when price_updating_or_errors reads max_price_spike.
                cleaning_config = get_config_for_price_filtering(data)
                update_historical_prices_for_instrument(
                    instrument_code,
                    data,
                    cleaning_config=cleaning_config,
                    interactive_mode=False,
                )
            downloaded_prices = True

        refreshed_multiple_adjusted = False
        if not options.dry_run:
            print("\nRefreshing multiple and adjusted prices")
            update_multiple_adjusted_prices_for_instrument(instrument_code, data)
            refreshed_multiple_adjusted = True
        else:
            print("\nDRY RUN: would refresh multiple and adjusted prices")

    return ReviveResult(
        starting_priced_contract=starting_priced_contract,
        ending_priced_contract=ending_priced_contract,
        rolls_performed=rolls_performed,
        refreshed_sampled_contracts=refreshed_sampled_contracts,
        downloaded_prices=downloaded_prices,
        refreshed_multiple_adjusted=refreshed_multiple_adjusted,
    )


def _roll_until_priced_contract_is_live(
    data: dataBlob,
    instrument_code: str,
    options: ReviveOptions,
) -> int:
    """Roll repeatedly until the priced contract's desired roll date is in future."""
    diag_prices = diagPrices(data)
    diag_contracts = dataContracts(data)
    rolls_performed = 0

    while True:
        current_contracts = diag_prices.get_multiple_prices(
            instrument_code
        ).current_contract_dict()

        priced_contract = current_contracts["PRICE"]
        forward_contract = current_contracts["FORWARD"]
        desired_roll_date = _desired_roll_date_for_priced_contract(
            diag_contracts,
            instrument_code,
            priced_contract,
        ).date()

        if desired_roll_date > options.as_of_date:
            print(
                f"Priced contract {priced_contract} rolls on {desired_roll_date}; "
                f"target date {options.as_of_date} is live."
            )
            return rolls_performed

        if rolls_performed >= options.max_rolls:
            raise RuntimeError(
                f"Refusing to roll {instrument_code} more than {options.max_rolls} "
                f"times; currently PRICE={priced_contract}, FORWARD={forward_contract}."
            )

        print(
            f"Rolling {instrument_code}: PRICE={priced_contract}, "
            f"FORWARD={forward_contract}, desired roll date {desired_roll_date}"
        )
        if options.dry_run:
            print("DRY RUN: would seed missing contract metadata and roll once")
            return rolls_performed

        # Direct manipulation is needed only because the interactive roll tool
        # asks for metadata for the stale priced contract before it gives the
        # user a chance to roll. Seeding missing metadata unblocks the standard
        # rolling object without changing existing metadata rows.
        _ensure_contracts_needed_for_next_roll_exist(
            diag_contracts=diag_contracts,
            instrument_code=instrument_code,
        )
        # allow_forward_fill is intentionally default-on for revive. Long-stale
        # series often end with missing carry/forward values, and the production
        # interactive flow asks this same question before rolling. This utility
        # makes that choice explicit and repeatable.
        rolling_prices = rollingAdjustedAndMultiplePrices(
            data,
            instrument_code,
            allow_forward_fill=options.allow_forward_fill,
        )
        rolling_prices.write_new_rolled_data()
        rolls_performed += 1


def _desired_roll_date_for_priced_contract(
    diag_contracts: dataContracts,
    instrument_code: str,
    priced_contract_id: str,
) -> datetime.datetime:
    """Return desired roll date even if stale contract metadata is missing."""
    try:
        return diag_contracts.when_to_roll_priced_contract(instrument_code)
    except ContractNotFound:
        # This is the exact stale-data failure revive is for: the multiple-price
        # row points at an old contract, but Mongo has no metadata row for it.
        # ContractDate can still compute the approximate expiry from YYYYMM00;
        # that is enough to decide that the instrument needs rolling.
        roll_parameters = diag_contracts.get_roll_parameters(instrument_code)
        contract_date_with_roll_parameters = contractDateWithRollParameters(
            contractDate(priced_contract_id), roll_parameters
        )
        return contract_date_with_roll_parameters.desired_roll_date


def _ensure_contracts_needed_for_next_roll_exist(
    diag_contracts: dataContracts,
    instrument_code: str,
) -> None:
    """Seed missing metadata rows required by one call to the roll calculator."""
    current_contracts = diag_contracts.get_current_contract_dict(instrument_code)
    contract_ids = {
        current_contracts.price,
        current_contracts.forward,
        current_contracts.carry,
    }

    # The next roll calculation reads metadata for the current FORWARD contract
    # and derives the next PRICE/FORWARD/CARRY contracts from roll parameters.
    # Add those derived contracts too, otherwise the following loop iteration
    # can fail for the same reason as the original interactive command.
    forward_contract_id = current_contracts.forward
    contract_ids.update(
        _next_contract_ids_after_forward_contract(
            diag_contracts,
            instrument_code,
            forward_contract_id,
        )
    )

    for contract_id in sorted(contract_ids):
        _ensure_contract_exists(diag_contracts, instrument_code, contract_id)


def _next_contract_ids_after_forward_contract(
    diag_contracts: dataContracts,
    instrument_code: str,
    forward_contract_id: str,
) -> set[str]:
    """Return contract ids the roll code will move to after ``forward_contract_id``."""
    roll_parameters = diag_contracts.get_roll_parameters(instrument_code)
    new_price_contract = contractDateWithRollParameters(
        contractDate(forward_contract_id), roll_parameters
    )
    return {
        new_price_contract.date_str,
        new_price_contract.next_held_contract().date_str,
        new_price_contract.carry_contract().date_str,
    }


def _ensure_contract_exists(
    diag_contracts: dataContracts,
    instrument_code: str,
    contract_id: str,
) -> None:
    """Create a minimal contract metadata row only when it is missing."""
    contract = futuresContract(futuresInstrument(instrument_code), contractDate(contract_id))
    if diag_contracts.is_contract_in_data(contract):
        return

    # Do not turn sampling on here. update_sampled_contracts owns the sampling
    # decision and will mark only the recent/live chain after IB expiry lookup.
    diag_contracts.add_contract_data(contract, ignore_duplication=False)
    print(f"Seeded missing contract metadata for {contract}")


def _current_priced_contract(diag_prices: diagPrices, instrument_code: str) -> str:
    multiple_prices = diag_prices.get_multiple_prices(instrument_code)
    return multiple_prices.current_contract_dict()["PRICE"]


def _parse_date(date_text: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(date_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected YYYY-MM-DD date, got {date_text!r}"
        ) from error


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll a stale futures instrument forward, refresh sampled contracts, "
            "download active contract prices, and update multiple/adjusted prices."
        )
    )
    parser.add_argument(
        "instrument",
        nargs="?",
        default=None,
        help="Instrument code to revive, e.g. COPPER-mini. Prompted if omitted.",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        default=datetime.date.today(),
        help="Roll until priced contract is live for this date. Default: today.",
    )
    parser.add_argument(
        "--max-rolls",
        type=int,
        default=48,
        help="Safety limit for repeated roll-adjusted operations. Default: 48.",
    )
    parser.add_argument(
        "--no-forward-fill",
        action="store_true",
        default=False,
        help="Do not forward-fill stale rows before each roll. Usually not useful for revive.",
    )
    parser.add_argument(
        "--skip-sampled-contracts",
        action="store_true",
        default=False,
        help="Do not run update_sampled_contracts after rolling.",
    )
    parser.add_argument(
        "--skip-price-download",
        action="store_true",
        default=False,
        help="Do not download/update sampled contract prices from IB.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show the first required roll without writing data or contacting IB.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    instrument_code = args.instrument
    if not instrument_code:
        with dataBlob(log_name="Revive-Instrument") as data:
            instrument_code = get_valid_instrument_code_from_user(data=data)

    if not instrument_code:
        print("No instrument code specified. Aborting.", file=sys.stderr)
        return 1

    if args.max_rolls < 1:
        print("--max-rolls must be >= 1", file=sys.stderr)
        return 2

    options = ReviveOptions(
        as_of_date=args.as_of,
        max_rolls=args.max_rolls,
        allow_forward_fill=not args.no_forward_fill,
        skip_sampled_contracts=args.skip_sampled_contracts,
        skip_price_download=args.skip_price_download,
        dry_run=args.dry_run,
    )

    try:
        result = revive_instrument(instrument_code.strip(), options)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 3

    print(
        "\nRevive complete: "
        f"{result.starting_priced_contract} -> {result.ending_priced_contract}; "
        f"rolls={result.rolls_performed}; "
        f"sampled_contracts={result.refreshed_sampled_contracts}; "
        f"downloaded_prices={result.downloaded_prices}; "
        f"multiple_adjusted={result.refreshed_multiple_adjusted}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
