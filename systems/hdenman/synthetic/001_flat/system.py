"""
systems/hdenman/synthetic/001_flat/system.py

Six-variant EWMAC system applied to SYN_FLAT.

    from systems.hdenman.synthetic.001_flat.system import flat_system
    system = flat_system()
    system.combForecast.get_combined_forecast("SYN_FLAT")
    system.positionSize.get_notional_position("SYN_FLAT")
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
    system = flat_system()
    print(system)
    print("instruments :", system.get_instrument_list())
    print()

    for rule in system.config.trading_rules:
        fc = system.forecastScaleCap.get_capped_forecast("SYN_FLAT", rule)
        print(f"  {rule:15s}  head={fc.dropna().head(1).values}  tail={fc.dropna().tail(1).values}")
