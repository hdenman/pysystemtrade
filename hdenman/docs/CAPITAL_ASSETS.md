# Capital & Leverage Management with Stock/Bond Collateral

## Code Architecture & Grounding

`pysystemtrade` manages production capital, brokerage account valuations, and leverage through several key layers:

1. **Broker Accounting Client (`sysbrokers/IB/client/ib_accounting_client.py`)**:
   - `broker_get_account_value_across_currency()` pulls the IB `NetLiquidation` tag across currencies.
   - `broker_get_excess_liquidity_value_across_currency()` pulls the IB `FullExcessLiquidity` tag.

2. **Broker Data Aggregator (`sysproduction/data/broker.py`)**:
   - `get_total_capital_value_in_base_currency()` converts all currency buckets of `NetLiquidation` into account base currency.
   - `get_margin_used_in_base_currency()` calculates total margin used as `NetLiquidation - FullExcessLiquidity`.

3. **Production Capital Tracker (`sysproduction/data/capital.py`, `sysproduction/update_total_capital.py`)**:
   - `update_total_capital()` fetches total account value (`NetLiquidation`) and updates the global capital records.
   - Strategy capital is allocated proportionally via `sysproduction/update_strategy_capital.py`.

4. **Capital Compounding Engine (`sysobjects/production/capital.py`)**:
   - Computes daily P&L as `new_broker_account_value - prev_broker_account_value`.
   - Compounding modes (`production_capital_method` in config):
     - `full`: Adds all P&L (positive or negative) directly to trading capital.
     - `half`: Losses reduce capital; profits restore capital only up to the high-water mark (`maximum_capital`).
     - `fixed`: Capital remains fixed regardless of account value changes.
   - Enforces a 10% safety check (`check_limit`): if daily account value changes by >10%, `LargeCapitalChange` is raised, triggering an error/email until manually inspected.

5. **Nightly Backtest & Position Sizing (`sysproduction/strategy_code/run_system_classic.py`, `systems/positionsizing.py`)**:
   - Nightly system runner reads strategy capital and overrides `config.notional_trading_capital`.
   - `PositionSizing.annual_cash_vol_target()` computes cash volatility target as `notional_trading_capital * percentage_vol_target / 100.0`.
   - Commodity contract position limits are scaled from this cash volatility target.

6. **System Risk Overlay & Leverage Limits (`systems/portfolio.py`, `systems/risk_overlay.py`)**:
   - Computes system leverage as sum of absolute portfolio weights relative to trading capital (`get_leverage_for_original_position()`).
   - Scales down positions if `risk_overlay.max_risk_leverage` limit is exceeded.

---

## Expected System Response for Commodities Traded with Stocks/Bonds as Collateral

If commodities are traded in a brokerage account where capital is held in stocks or bonds instead of cash:

### 1. Trading Capital Derivation
`pysystemtrade` uses IB `NetLiquidation` as total account capital. It does not isolate cash from securities. If the account holds $1,000,000 in stocks/bonds and minimal cash, `pysystemtrade` sees $1,000,000 as available trading capital.

### 2. Collateral Mark-to-Market Impact
Daily market fluctuations in the stock/bond portfolio change the account `NetLiquidation` and are interpreted by the system as account P&L:
- **`full` compounding**: Stock/bond price gains increase commodity position sizes; stock/bond losses decrease commodity position sizes.
- **`half` compounding**: Stock/bond losses reduce commodity position sizes. Stock/bond gains increase commodity position sizes only up to the high-water mark.
- **`fixed` compounding**: Stock/bond price moves update reported account valuation but leave futures trading capital unchanged.

### 3. Safety Thresholds & Transfers
- **Large Price Moves**: If stock/bond collateral moves by >10% in a single update, `update_total_capital` raises `LargeCapitalChange` and halts capital updates until confirmed manually via `interactive_update_capital_manual`.
- **Securities Inflow/Outflow**: Depositing or withdrawing stock/bond certificates appears to `pysystemtrade` as a trading P&L event. Use the interactive menu ("Adjust account value for withdrawal or deposit") to prevent transfers from altering trading capital.

### 4. Broker Margin vs. System Exposure Limits
- **Broker Margin**: IB applies haircuts and initial/maintenance margin rules to stock/bond collateral. Margin used is recorded as `NetLiquidation - FullExcessLiquidity`.
- **System Exposure**: The system's position sizing and leverage overlays (`max_risk_leverage`) use `NetLiquidation` as the denominator. They do not monitor cash buffer requirements or collateral haircuts. If collateral value drops or haircuts increase, IB may issue margin calls or liquidate futures positions before system risk checks trigger.

### 5. Unmodeled Risk Overlay Gap
The stock and bond holdings in the broker account are not tracked as positions within the futures `System` model. Consequently:
- Asset volatility and correlation between collateral assets (e.g., bonds/equities) and commodities are ignored in `systems/risk_overlay.py`.
- The commodity strategy will execute its target annual volatility assuming the entire capital base is dedicated solely to the commodity futures program.

---

## Recommended Configuration Adjustments

1. **Manual Capital Initialization**: Set `notional_trading_capital` or override total capital to represent only the target capital allocated to commodity futures, rather than full account `NetLiquidation`.
2. **Use Fixed/Half Compounding**: Set `production_capital_method: half` or `fixed` in YAML to avoid compounding non-commodity portfolio noise into futures position sizing.
3. **Cash & Margin Monitoring**: Maintain a dedicated cash buffer for commodity futures margin/variation sweeps, as `pysystemtrade` does not calculate cash interest or collateral liquidity haircuts automatically.
