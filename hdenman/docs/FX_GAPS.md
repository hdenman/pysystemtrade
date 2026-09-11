# FX Price Data Sources and Gap Resolution

This document summarizes how `pysystemtrade` manages FX/forex data, why gaps occur (e.g., in `EURUSD` / `USDEUR`), and how to resolve them.

---

## 1. Where the System Gets FX Data

`pysystemtrade` uses a two-tier architecture for currency exchange rates:

### A. Seed / Bootstrap Data (CSV Files)
- **Location:** `data/futures/fx_prices_csv/` (e.g., `EURUSD.csv`, `GBPUSD.csv`, `JPYUSD.csv`).
- **Convention:** All rates are quoted against USD as `CCYUSD` (e.g., `EURUSD` represents the price of 1 EUR in USD).
- **Initial Load Script:** `sysinit/futures/repocsv_spotfx_prices.py` writes these CSV files into your persistent database (Arctic/MongoDB/Parquet).

### B. Live Daily Production Updates (Interactive Brokers)
- **Script:** `sysproduction/linux/scripts/update_fx_prices` (typically run via cron).
- **Module:** `sysbrokers.IB.ib_Fx_prices_data.ibFxPricesData`
- **Configuration:** `sysbrokers/IB/ibConfigSpotFX.csv`
- **Limitation:** Interactive Brokers generally returns only **up to 1 year of daily historical prices** for spot FX pairs.

---

## 2. Why Gaps Occur in USDEUR / EURUSD

1. **Interactive Brokers Lookback Limit:** If your database was initialized with stale or gapped seed data and IB updates are run, IB will not backfill missing history older than 1 year.
2. **Missing Market Closures in Rules:** Business days missed due to unknown holidays or unpopulated periods in the initial seed files will trigger `util/check_instrument` health warnings.
3. **Naming Convention:** `pysystemtrade` tracks `EURUSD` (USD per EUR) rather than `USDEUR`.

---

## 3. How to Resolve FX Data Gaps

1. **Update the Seed CSV:**
   Edit or overwrite `data/futures/fx_prices_csv/EURUSD.csv` with complete historical data (with `DATETIME,PRICE` header and standard `YYYY-MM-DD HH:MM:SS,RATE` rows). External historical data can be sourced from providers such as Investing.com or Quandl.

2. **Re-seed the Production Database:**
   Run the seeding script to update your local database from the revised CSV:
   ```bash
   python -m sysinit.futures.repocsv_spotfx_prices
   ```
   Or use `sysinit/futures/spotfx_from_csvAndInvestingDotCom_to_db.py` to import custom external CSV downloads directly.

3. **Verify with Health Check:**
   Validate that business-day gaps have been resolved:
   ```bash
   python -m util.check_instrument EUR
   ```
