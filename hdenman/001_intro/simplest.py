import pandas as pd
from sysquant.estimators.vol import robust_vol_calc
from sysdata.sim.csv_futures_sim_data import csvFuturesSimData

from matplotlib.pyplot import show

def calc_ewmac_forecast(price, Lfast, Lslow=None):
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds Lfast, Lslow, and vol_lookback

    """

    price = price.resample("1B").last()
    if Lslow is None:
        Lslow = Lfast * 4

    fast_ewma = price.ewm(span=Lfast).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma

    vol = robust_vol_calc(price.diff())

    return raw_ewmac / vol


def get_data():
    data = csvFuturesSimData()
    return data

def main():
    data = get_data()

    instrument_code = 'SOFR'
    price=data.daily_prices(instrument_code)
    ewmac = calc_ewmac_forecast(price, 32, 128)
    ewmac.tail(5)

    ewmac.plot()
    show()


if __name__ == "__main__":
    main()
