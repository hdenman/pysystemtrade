"""
Clear all orders from the active universe's order stacks.

Deactivates then removes every order on the instrument, contract, and broker
stacks.  Safe to run at any time the stack handler is not actively executing
(i.e. outside the 00:15–23:50 run_stack_handler window, or before IB Gateway
is connected).

WARNING: this bypasses the normal safe_stack_removal path — it does NOT cancel
open broker orders with IB first, and it does NOT archive orders to the historic
log.  Use interactively only, not as a scheduled task.

Run as a module::

    PYSYS_UNIVERSE=futures python -m util.clear_orders
    PYSYS_UNIVERSE=futures python -m util.clear_orders --confirm
"""

import argparse
import sys

from sysdata.data_blob import dataBlob
from sysproduction.data.orders import dataOrders
from syscore.universe import get_universe


def clear_all_stacks(data: dataOrders) -> dict[str, int]:
    """Deactivate and remove every order on all three stacks.

    :return: dict of stack_name → count of orders removed
    """
    results = {}
    for stack_name, stack in [
        ("instrument", data.db_instrument_stack_data),
        ("contract",   data.db_contract_stack_data),
        ("broker",     data.db_broker_stack_data),
    ]:
        ids = stack.get_list_of_order_ids(exclude_inactive_orders=False)
        for oid in ids:
            stack.deactivate_order(oid)
            stack.remove_order_with_id_from_stack(oid)
        results[stack_name] = len(ids)
    return results


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clear all orders from the active universe's order stacks. "
            "Does NOT cancel open IB broker orders first — use only when "
            "the stack handler is not running."
        )
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Skip the interactive confirmation prompt.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    universe = get_universe().value

    if not args.confirm:
        print(f"\nUniverse : {universe}")
        print("This will deactivate and remove ALL orders from all three stacks.")
        print("It does NOT cancel broker orders with IB or archive to history.")
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 0

    with dataBlob(log_name="clear_orders") as blob:
        data = dataOrders(blob)
        results = clear_all_stacks(data)

    total = sum(results.values())
    for stack_name, count in results.items():
        print(f"  {stack_name:12s}: {count} order(s) removed")
    print(f"  {'total':12s}: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
