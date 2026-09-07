"""
List open contract positions directly from Interactive Brokers (IB).

Uses dataBlob(log_name="list_positions") and sysproduction.data.broker.dataBroker
(the same underlying data layer as interactive_order_stack).

Run as a module:

    python -m util.list_positions
    python -m util.list_positions --account <ACCOUNT_ID>
"""

import argparse
import sys
import pandas as pd

from sysdata.data_blob import dataBlob
from sysproduction.data.broker import dataBroker


def get_ib_positions(data: dataBlob, account_id: str | None = None) -> pd.DataFrame:
    """Fetch current contract positions from IB using dataBroker."""
    data_broker = dataBroker(data)
    if account_id is not None:
        list_of_positions = (
            data_broker.broker_contract_position_data.get_all_current_positions_as_list_with_contract_objects(
                account_id=account_id
            )
        )
    else:
        list_of_positions = data_broker.get_all_current_contract_positions()

    df = list_of_positions.as_pd_df()
    if df.empty:
        return df

    if "instrument_code" in df.columns and "contract_date" in df.columns:
        df = df.sort_values(["instrument_code", "contract_date"]).reset_index(drop=True)

    return df


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display open positions from Interactive Brokers (IB)."
    )
    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help="Optional IB account ID filter (defaults to configured broker account).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    with dataBlob(log_name="list_positions") as blob:
        df = get_ib_positions(blob, account_id=args.account)
        if df.empty:
            print("No open positions found on IB.")
        else:
            print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
