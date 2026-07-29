"""
This is a futures system

A system consists of a system, plus a config

"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from syscore.constants import arg_not_supplied

from sysdata.sim.csv_futures_sim_data import csvFuturesSimData
from sysdata.sim.db_futures_sim_data import dbFuturesSimData
from sysdata.config.configdata import Config

from systems.forecasting import Rules
from systems.basesystem import System
from systems.forecast_combine import ForecastCombine
from systems.forecast_scale_cap import ForecastScaleCap
from systems.rawdata import RawData
from systems.positionsizing import PositionSizing
from systems.portfolio import Portfolios
from systems.accounts.accounts_stage import Account


def futures_system(
    data=arg_not_supplied,
    config=arg_not_supplied,
    trading_rules=arg_not_supplied,
):
    """

    :param data: data object (defaults to reading from csv files)
    :type data: sysdata.data.simData, or anything that inherits from it

    :param config: Configuration object (defaults to futuresconfig.yaml in this directory)
    :type config: sysdata.configdata.Config

    :param trading_rules: Set of trading rules to use (defaults to set specified in config object)
    :type trading_rules: list or dict of TradingRules, or something that can be parsed to that


    >>> system=futures_system()
    >>> system
    System with stages: accounts, portfolio, positionSize, rawdata, combForecast, forecastScaleCap, rules
    >>> system.rules.get_raw_forecast("EDOLLAR", "ewmac2_8").dropna().head(2)
                ewmac2_8
    1983-10-10  0.695929
    1983-10-11 -0.604704

                ewmac2_8
    2015-04-21  0.172416
    2015-04-22 -0.477559
    >>> system.rules.get_raw_forecast("EDOLLAR", "carry").dropna().head(2)
                   carry
    1983-10-10  0.952297
    1983-10-11  0.854075

                   carry
    2015-04-21  0.350892
    2015-04-22  0.350892
    """

    if data is arg_not_supplied:
        data = csvFuturesSimData()

    if config is arg_not_supplied:
        config = Config("systems.hdenman.backtest_1.yaml")

    rules = Rules(trading_rules)

    system = System(
        [
            Account(),
            Portfolios(),
            PositionSizing(),
            RawData(),
            ForecastCombine(),
            ForecastScaleCap(),
            rules,
        ],
        data,
        config,
    )

    return system


if __name__ == "__main__":
    import doctest

    doctest.testmod()


def plot_forecast_vs_price(
    system,
    instrument_code: str,
    figsize: tuple = (14, 5),
) -> tuple:
    """
    Plot combined forecast and daily price on dual y-axes.

    Returns (fig, ax_price, ax_forecast) for further customisation.
    """
    price = system.rawdata.get_daily_prices(instrument_code)
    forecast = system.combForecast.get_combined_forecast(instrument_code)

    # Trim to the forecast's valid (post-warmup) window; forward-fill price to
    # the same index so both series share identical date ticks.
    forecast = forecast.dropna()
    price = price.reindex(forecast.index, method="ffill")

    fig, ax_price = plt.subplots(figsize=figsize)
    ax_fc = ax_price.twinx()

    l_price, = ax_price.plot(
        price.index, price.values,
        color="steelblue", lw=1.2, label="Price",
    )
    l_fc, = ax_fc.plot(
        forecast.index, forecast.values,
        color="darkorange", lw=0.9, alpha=0.85, label="Combined forecast",
    )
    ax_fc.axhline(0, color="darkorange", lw=0.5, ls="--", alpha=0.4)

    ax_price.set_ylabel("Price", color="steelblue")
    ax_price.tick_params(axis="y", colors="steelblue")
    ax_fc.set_ylabel("Combined forecast", color="darkorange")
    ax_fc.tick_params(axis="y", colors="darkorange")

    # Merge handles from both axes into a single legend on the left axis.
    ax_price.legend(handles=[l_price, l_fc], loc="upper left")

    ax_price.set_title(f"{instrument_code}: price vs combined forecast")
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax_price, ax_fc


def plot_price_vs_position(
    system,
    instrument_code: str,
    figsize: tuple = (14, 5),
) -> tuple:
    """
    Plot notional portfolio position and daily price on dual y-axes.

    Returns (fig, ax_price, ax_position) for further customisation.
    """
    price = system.rawdata.get_daily_prices(instrument_code)
    position = system.portfolio.get_notional_position(instrument_code)

    position = position.dropna()
    price = price.reindex(position.index, method="ffill")

    fig, ax_price = plt.subplots(figsize=figsize)
    ax_pos = ax_price.twinx()

    l_price, = ax_price.plot(
        price.index, price.values,
        color="steelblue", lw=1.2, label="Price",
    )
    l_pos, = ax_pos.plot(
        position.index, position.values,
        color="seagreen", lw=0.9, alpha=0.85, label="Notional position",
    )
    ax_pos.axhline(0, color="seagreen", lw=0.5, ls="--", alpha=0.4)

    ax_price.set_ylabel("Price", color="steelblue")
    ax_price.tick_params(axis="y", colors="steelblue")
    ax_pos.set_ylabel("Notional position (contracts)", color="seagreen")
    ax_pos.tick_params(axis="y", colors="seagreen")

    ax_price.legend(handles=[l_price, l_pos], loc="upper left")
    ax_price.set_title(f"{instrument_code}: price vs notional position")
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax_price, ax_pos
