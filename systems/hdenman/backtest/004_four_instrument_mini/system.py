"""
systems/hdenman/backtest/004_four_instrument_mini/system.py

Six-variant EWMAC system applied to SP500_micro, GOLD_micro, US10, GAS_US_mini.

Smaller-contract equivalent of 003_four_instrument; same four asset classes
(Equity / Metal / Bond / Energy) with micro/mini-sized contracts.

Instrument metadata comes from the global instrumentconfig.csv.
Prices come from the backtest parquet store (loaded by setup.py).

Usage
-----
    python systems/hdenman/backtest/004_four_instrument_mini/setup.py   # once
    python systems/hdenman/backtest/004_four_instrument_mini/system.py  # run → report
"""

import os

os.environ["PYSYS_UNIVERSE"] = "backtest"

from syscore.constants import arg_not_supplied
from sysdata.config.configdata import Config
from sysdata.sim.db_futures_sim_data import dbFuturesSimData

from systems.basesystem import System
from systems.forecasting import Rules
from systems.rawdata import RawData
from systems.forecast_scale_cap import ForecastScaleCap
from systems.forecast_combine import ForecastCombine
from systems.positionsizing import PositionSizing
from systems.portfolio import Portfolios
from systems.accounts.accounts_stage import Account


def four_instrument_mini_system(
    data=arg_not_supplied,
    config=arg_not_supplied,
) -> System:
    """
    SP500_micro + GOLD_micro + US10 + GAS_US_mini EWMAC backtest.

    Uses the global instrumentconfig.csv for instrument metadata —
    no local csv_data_paths override needed.
    """
    if data is arg_not_supplied:
        data = dbFuturesSimData()

    if config is arg_not_supplied:
        config = Config("systems.hdenman.backtest.004_four_instrument_mini.system.yaml")

    return System(
        [
            Account(),
            Portfolios(),
            PositionSizing(),
            RawData(),
            ForecastCombine(),
            ForecastScaleCap(),
            Rules(),
        ],
        data,
        config,
    )


if __name__ == "__main__":
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

    from util.backtest import run_backtest

    system = four_instrument_mini_system()
    path = run_backtest(system, name="004_four_instrument_mini", starting_capital=1_000_000)
    print(f"Report: {path}")
    subprocess.run(["open", str(path)])
