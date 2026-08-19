"""
systems/hdenman/synthetic/003_noisy_sine/system.py

Six-variant EWMAC system applied to SYN_NOISY_SINE.

Usage
-----
    python systems/hdenman/synthetic/003_noisy_sine/system.py
"""

import os

os.environ["PYSYS_UNIVERSE"] = "synthetic"

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

_INSTRUMENT_CONFIG_PATH = "systems.hdenman.synthetic.003_noisy_sine"


def noisy_sine_system(
    data=arg_not_supplied,
    config=arg_not_supplied,
) -> System:
    if data is arg_not_supplied:
        data = dbFuturesSimData(
            csv_data_paths={
                "csvFuturesInstrumentData": _INSTRUMENT_CONFIG_PATH,
            }
        )
    if config is arg_not_supplied:
        config = Config("systems.hdenman.synthetic.003_noisy_sine.system.yaml")

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
    import subprocess, sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

    from util.backtest import run_backtest

    system = noisy_sine_system()
    path = run_backtest(system, name="003_noisy_sine", starting_capital=1_000_000)
    print(f"Report: {path}")
    subprocess.run(["open", str(path)])
