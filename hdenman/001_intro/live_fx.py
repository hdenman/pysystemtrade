from sysbrokers.IB.ib_connection import connectionIB
from sysbrokers.IB.ib_Fx_prices_data import ibFxPricesData
from sysdata.data_blob import dataBlob

from systems.accounts.account_forecast import pandl_for_instrument_forecast


import pandas as pd
import matplotlib.pyplot as plt
from sysquant.estimators.vol import robust_vol_calc


def calc_ewmac_forecast(price, Lfast, Lslow=None):
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback

    """
    if Lslow is None:
        Lslow = 4 * Lfast

    ## We don't need to calculate the decay parameter, just use the span directly
    fast_ewma = price.ewm(span=Lfast).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma

    vol = robust_vol_calc(price.diff())

    return raw_ewmac / vol



def plot(price, ewmac):
    plt.figure(figsize=(12,5))

    ax1 = price.plot(color='blue', grid=True, label='Price')
    ax2 = ewmac.plot(color='red', grid=True, secondary_y=True, label='Forecast')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()


    plt.legend(h1+h2, l1+l2, loc=2)
    plt.show()


# conn = connectionIB(111, ib_port=4001)
def run(conn, span=32):
    ibfxpricedata = ibFxPricesData(conn, dataBlob())
    price=ibfxpricedata['EURUSD']
    ewmac=calc_ewmac_forecast(price, span)
    plot(price, ewmac)
    account = pandl_for_instrument_forecast(forecast = ewmac, price = price)
    print(account.percent.stats())
