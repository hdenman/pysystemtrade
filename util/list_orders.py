"""
List all orders and their status for the active universe.

Three stacks are shown:

* **Instrument orders** — strategy-level desired positions (one per instrument).
* **Contract orders** — translated to specific futures contracts.
* **Broker orders** — individual fills sent to (or returned from) IB.

Plus the two historic order logs (strategy and broker), which hold completed /
archived orders from previous cycles.

The active universe is controlled by ``PYSYS_UNIVERSE`` (default: synthetic).
MongoDB is already universe-aware (database = ``production_<universe>``).

Run as a module::

    python -m util.list_orders
    python -m util.list_orders --stack instrument
    python -m util.list_orders --stack contract
    python -m util.list_orders --stack broker
    python -m util.list_orders --stack historic_instrument
    python -m util.list_orders --stack historic_broker
    python -m util.list_orders --all-stacks
    python -m util.list_orders --days 7          # historic: last N days
"""

import argparse
import datetime
import sys

import pandas as pd

from syscore.constants import arg_not_supplied
from sysdata.data_blob import dataBlob
from sysexecution.orders.named_order_objects import no_children, no_parent
from sysproduction.data.orders import dataOrders

# ── constants ──────────────────────────────────────────────────────────────────

_STACK_CHOICES = [
    "instrument",
    "contract",
    "broker",
    "historic_instrument",
    "historic_broker",
]

_DEFAULT_HISTORIC_DAYS = 1


# ── stack reporters ────────────────────────────────────────────────────────────


def _fmt_qty(qty) -> str:
    """tradeQuantity → compact string, e.g. '[3]' or '[6, -6]'."""
    lst = list(qty)
    if len(lst) == 1:
        return str(lst[0])
    return str(lst)


def _status(order) -> str:
    """One-word status derived from order state flags.

    There is no dedicated cancelled flag in pysystemtrade — an order that was
    sent to IB and then cancelled before any fill is indistinguishable from one
    that was deactivated without ever being submitted.  Both land as inactive
    with fill == 0, which we label 'cancelled' here to distinguish from a
    fully-filled inactive order.
    """
    if not order.active:
        if order.fill_equals_desired_trade():
            return "filled"
        if not order.fill_equals_zero():
            return "partial"
        return "cancelled"   # inactive, zero fill — never executed
    if order._locked:
        return "locked"
    if not order.fill_equals_zero():
        if order.fill_equals_desired_trade():
            return "filled"
        return "partial"
    return "open"


def _children_str(order) -> str:
    if order.children is no_children:
        return "—"
    return str(order.children)


def _parent_str(order) -> str:
    if order.parent is no_parent:
        return "—"
    return str(order.parent)


# ── instrument stack ───────────────────────────────────────────────────────────


def report_instrument_stack(data: dataOrders) -> pd.DataFrame:
    """Print and return all orders on the instrument order stack."""
    stack = data.db_instrument_stack_data
    ids_active = stack.get_list_of_order_ids(exclude_inactive_orders=False)

    rows = []
    for oid in sorted(ids_active):
        o = stack.get_order_with_id_from_stack(oid)
        rows.append(
            {
                "id": o.order_id,
                "strategy": o.strategy_name,
                "instrument": o.instrument_code,
                "type": str(o.order_type),
                "trade": _fmt_qty(o.trade),
                "fill": _fmt_qty(o.fill),
                "fill_price": o.filled_price,
                "fill_dt": _fmt_dt(o.fill_datetime),
                "generated": _fmt_dt(o.generated_datetime),
                "status": _status(o),
                "parent": _parent_str(o),
                "children": _children_str(o),
            }
        )

    df = pd.DataFrame(rows) if rows else _empty_df(
        ["id", "strategy", "instrument", "type", "trade", "fill",
         "fill_price", "fill_dt", "generated", "status", "parent", "children"]
    )
    _print_section("INSTRUMENT ORDER STACK", df)
    return df


# ── contract stack ─────────────────────────────────────────────────────────────


def report_contract_stack(data: dataOrders) -> pd.DataFrame:
    """Print and return all orders on the contract order stack."""
    stack = data.db_contract_stack_data
    ids_active = stack.get_list_of_order_ids(exclude_inactive_orders=False)

    rows = []
    for oid in sorted(ids_active):
        o = stack.get_order_with_id_from_stack(oid)
        rows.append(
            {
                "id": o.order_id,
                "strategy": o.strategy_name,
                "instrument": o.instrument_code,
                "contract": str(o.tradeable_object.contract_date),
                "type": str(o.order_type),
                "trade": _fmt_qty(o.trade),
                "fill": _fmt_qty(o.fill),
                "fill_price": o.filled_price,
                "ref_price": o.order_info.get("reference_price"),
                "status": _status(o),
                "parent": _parent_str(o),
                "children": _children_str(o),
            }
        )

    df = pd.DataFrame(rows) if rows else _empty_df(
        ["id", "strategy", "instrument", "contract", "type", "trade", "fill",
         "fill_price", "ref_price", "status", "parent", "children"]
    )
    _print_section("CONTRACT ORDER STACK", df)
    return df


# ── broker stack ───────────────────────────────────────────────────────────────


def report_broker_stack(data: dataOrders) -> pd.DataFrame:
    """Print and return all orders on the broker order stack."""
    stack = data.db_broker_stack_data
    ids_active = stack.get_list_of_order_ids(exclude_inactive_orders=False)

    rows = []
    for oid in sorted(ids_active):
        o = stack.get_order_with_id_from_stack(oid)
        rows.append(
            {
                "id": o.order_id,
                "instrument": o.instrument_code,
                "contract": str(o.tradeable_object.contract_date),
                "type": str(o.order_type),
                "trade": _fmt_qty(o.trade),
                "fill": _fmt_qty(o.fill),
                "fill_price": o.filled_price,
                "broker_order_id": o.order_info.get("broker_order_id", "—"),
                "status": _status(o),
                "parent": _parent_str(o),
            }
        )

    df = pd.DataFrame(rows) if rows else _empty_df(
        ["id", "instrument", "contract", "type", "trade", "fill",
         "fill_price", "broker_order_id", "status", "parent"]
    )
    _print_section("BROKER ORDER STACK", df)
    return df


# ── historic orders ────────────────────────────────────────────────────────────


def _date_range(days: int) -> tuple[datetime.datetime, datetime.datetime]:
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    return start, end


def report_historic_instrument_orders(
    data: dataOrders, days: int = _DEFAULT_HISTORIC_DAYS
) -> pd.DataFrame:
    """Print and return historic instrument orders from the last *days* days."""
    start, end = _date_range(days)
    ids = data.get_historic_instrument_order_ids_in_date_range(start, end)

    rows = []
    for oid in sorted(ids):
        o = data.get_historic_instrument_order_from_order_id(oid)
        rows.append(
            {
                "id": o.order_id,
                "strategy": o.strategy_name,
                "instrument": o.instrument_code,
                "type": str(o.order_type),
                "trade": _fmt_qty(o.trade),
                "fill": _fmt_qty(o.fill),
                "fill_price": o.filled_price,
                "fill_dt": _fmt_dt(o.fill_datetime),
                "generated": _fmt_dt(o.generated_datetime),
                "status": _status(o),
            }
        )

    df = pd.DataFrame(rows) if rows else _empty_df(
        ["id", "strategy", "instrument", "type", "trade", "fill",
         "fill_price", "fill_dt", "generated", "status"]
    )
    _print_section(
        f"HISTORIC INSTRUMENT ORDERS (last {days}d: {start:%Y-%m-%d} → {end:%Y-%m-%d})",
        df,
    )
    return df


def report_historic_broker_orders(
    data: dataOrders, days: int = _DEFAULT_HISTORIC_DAYS
) -> pd.DataFrame:
    """Print and return historic broker orders from the last *days* days."""
    start, end = _date_range(days)
    ids = data.get_historic_broker_order_ids_in_date_range(start, end)

    rows = []
    for oid in sorted(ids):
        o = data.get_historic_broker_order_from_order_id(oid)
        rows.append(
            {
                "id": o.order_id,
                "instrument": o.instrument_code,
                "contract": str(o.tradeable_object.contract_date),
                "type": str(o.order_type),
                "trade": _fmt_qty(o.trade),
                "fill": _fmt_qty(o.fill),
                "fill_price": o.filled_price,
                "fill_dt": _fmt_dt(o.fill_datetime),
                "broker_order_id": o.order_info.get("broker_order_id", "—"),
                "status": _status(o),
            }
        )

    df = pd.DataFrame(rows) if rows else _empty_df(
        ["id", "instrument", "contract", "type", "trade", "fill",
         "fill_price", "fill_dt", "broker_order_id", "status"]
    )
    _print_section(
        f"HISTORIC BROKER ORDERS (last {days}d: {start:%Y-%m-%d} → {end:%Y-%m-%d})",
        df,
    )
    return df


# ── formatting helpers ─────────────────────────────────────────────────────────


def _fmt_dt(dt) -> str:
    if dt is None:
        return "—"
    if isinstance(dt, datetime.datetime):
        return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)


def _empty_df(columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _print_section(title: str, df: pd.DataFrame) -> None:
    bar = "─" * max(len(title) + 4, 60)
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    if df.empty:
        print("  (none)")
    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        pd.set_option("display.max_colwidth", 20)
        print(df.to_string(index=False))


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List orders and their status for the active PYSYS_UNIVERSE."
    )
    parser.add_argument(
        "--stack",
        choices=_STACK_CHOICES,
        default=None,
        metavar="STACK",
        help=(
            "Which stack to show. One of: "
            + ", ".join(_STACK_CHOICES)
            + ". Default: show instrument + contract + broker stacks."
        ),
    )
    parser.add_argument(
        "--all-stacks",
        action="store_true",
        default=False,
        help="Show every stack including historic orders.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=_DEFAULT_HISTORIC_DAYS,
        metavar="N",
        help=f"Days back for historic order queries (default: {_DEFAULT_HISTORIC_DAYS}).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    with dataBlob(log_name="list_orders") as blob:
        data = dataOrders(blob)

        if args.stack is not None:
            # single explicit stack
            {
                "instrument": lambda: report_instrument_stack(data),
                "contract": lambda: report_contract_stack(data),
                "broker": lambda: report_broker_stack(data),
                "historic_instrument": lambda: report_historic_instrument_orders(
                    data, days=args.days
                ),
                "historic_broker": lambda: report_historic_broker_orders(
                    data, days=args.days
                ),
            }[args.stack]()

        elif args.all_stacks:
            report_instrument_stack(data)
            report_contract_stack(data)
            report_broker_stack(data)
            report_historic_instrument_orders(data, days=args.days)
            report_historic_broker_orders(data, days=args.days)

        else:
            # default: live stacks only
            report_instrument_stack(data)
            report_contract_stack(data)
            report_broker_stack(data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
