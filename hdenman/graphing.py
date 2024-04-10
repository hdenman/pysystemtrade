import matplotlib.pyplot as plt
import numpy as np
from sysquant.estimators.vol import robust_vol_calc
import pandas as pd

# Replicating code from systems.provided.rules.ewmac.calc_ewmac_forecast(price, fast, slow))
# for learning purposes


# Create some mock data
def show_ewmac(price: pd.Series, fast: int = 32, slow: int = 128, vol=True) -> None:
    # Price times (day granularity)
    t = np.array(price.index)

    # 'ewm' is a pd.Series method.  Decay param: alpha = 2 / (span + 1)
    # 'ewm' returns an ExponentialMovingWindow; hence 'mean' afterwards
    fast_ewm = price.ewm(span=fast, min_periods=1).mean()
    slow_ewm = price.ewm(span=slow, min_periods=1).mean()

    ewmac = fast_ewm - slow_ewm
    if vol:
        vol_days = 35
        vol = robust_vol_calc(price, vol_days)
        # ffil: replace NaNs with previous good value
        ewmac = ewmac / vol.ffill()

    price = np.array(price)
    fast_ewm = np.array(fast_ewm)
    slow_ewm = np.array(slow_ewm)
    ewmac = np.array(ewmac)

    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("price", color=color)
    ax1.plot(t, price, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    color = "tab:green"
    ax1.plot(t, fast_ewm, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    color = "tab:pink"
    ax1.plot(t, slow_ewm, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("ewmac", color=color)  # we already handled the x-label with ax1
    ax2.plot(t, ewmac, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
