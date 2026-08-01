"""
systems/hdenman/synthetic/001_flat/system.py

Six-variant EWMAC system applied to SYN_FLAT.

    from systems.hdenman.synthetic.001_flat.system import flat_system
    system = flat_system()
    system.combForecast.get_combined_forecast("SYN_FLAT")
    system.positionSize.get_notional_position("SYN_FLAT")

or

    python systems/hdenman/synthetic/001_flat/setup.py
    python systems/hdenman/synthetic/001_flat/system.py

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

# Instrument config lives alongside this file, not in the global csvconfig.
_INSTRUMENT_CONFIG_PATH = "systems.hdenman.synthetic.001_flat"


def flat_system(
    data=arg_not_supplied,
    config=arg_not_supplied,
) -> System:
    """
    Build the SYN_FLAT EWMAC system.

    Parameters
    ----------
    data   : override the default dbFuturesSimData (useful for testing)
    config : override the default system.yaml (useful for parameter sweeps)
    """
    if data is arg_not_supplied:
        data = dbFuturesSimData(
            csv_data_paths={
                "csvFuturesInstrumentData": _INSTRUMENT_CONFIG_PATH,
            }
        )

    if config is arg_not_supplied:
        config = Config("systems.hdenman.synthetic.001_flat.system.yaml")

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
    # system.py lives 5 dirs deep; climb to project root so 'util' is importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

    from util.backtest import run_backtest

    system = flat_system()
    path = run_backtest(system, name="001_flat", starting_capital=1_000_000)
    print(f"Report: {path}")
    subprocess.run(["open", str(path)])
