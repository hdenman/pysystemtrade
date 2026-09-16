# System data storage and order pipeline

Production data is wired in `sysproduction/data/production_data_objects.py`. Price-like time series, positions, and backtest targets use Parquet by default; contract metadata, order stacks, and process control use MongoDB; instrument config and roll parameters come from CSV.

Parquet root is `PARQUET_DATA` if set, otherwise config key `parquet_store` (`sysdata/data_blob.py`). Each Parquet data type is a subdirectory; each identifier is one `.parquet` file.

## Storage types

| Data | Interface / production class | Physical key / collection | Shape |
|---|---|---|---|
| Sampled contracts | `dataContracts` -> `mongoFuturesContractData` | Mongo collection `futures_contracts`, key `contract_key = INSTRUMENT/CONTRACT` | `futuresContract.as_dict()`: instrument, contract date/expiry fields, `contract_params.sampling` |
| Traded contracts | `diagPositions` / `updatePositions` -> `parquetContractPositionData` | `contract_positions/INSTRUMENT#CONTRACT.parquet` | Time series column `position`; current traded contracts are files whose last position is non-zero |
| Strategy positions | `diagPositions` -> `parquetStrategyPositionData` | `strategy_positions/STRATEGY_INSTRUMENT.parquet` | Time series column `position` per strategy and instrument |
| Optimal positions | `dataOptimalPositions` -> `parquetOptimalPositionData` | `optimal_positions/STRATEGY_INSTRUMENT.parquet` | Row indexed by datetime with `lower_position`, `upper_position`, `reference_price`, `reference_contract` |
| Backtest state | `dataBacktest` -> `sysproduction/data/backtest.py` | `backtest_store_directory/STRATEGY/YYYYMMDD_HHMMSS_backtest.pckz` | Pickled `System` object containing backtest object graph |
| Order stacks | `dataOrders` -> `mongoInstrumentOrderStackData`, `mongoContractOrderStackData`, `mongoBrokerOrderStackData` | Mongo collections `INSTRUMENT_ORDER_STACK`, `CONTRACT_ORDER_STACK`, `BROKER_ORDER_STACK` | Document dicts representing pending/active orders at each execution abstraction layer |
| Historic orders | `dataOrders` -> `mongoStrategyHistoricOrdersData`, `mongoContractHistoricOrdersData`, `mongoBrokerHistoricOrdersData` | Mongo collections `strategy_historic_orders`, `contract_historic_orders`, `broker_historic_orders` | Execution history records for audit |
| Per-contract prices | `diagPrices` / `updatePrices` -> `parquetFuturesContractPriceData` | `futures_contract_prices/[FREQ@]INSTRUMENT#CONTRACT.parquet` | DataFrame with `OPEN`, `HIGH`, `LOW`, `FINAL`, `VOLUME` |
| Multiple prices | `futuresMultiplePricesData` -> `parquetFuturesMultiplePricesData` | `futures_multiple_prices/INSTRUMENT.parquet` | DataFrame containing `PRICE`, `FORWARD`, `CARRY` plus their contract-id columns |
| Adjusted prices | `futuresAdjustedPricesData` -> `parquetFuturesAdjustedPricesData` | `futures_adjusted_prices/INSTRUMENT.parquet` | Single continuous series column `price` |
| FX prices | `dataCurrency` -> `parquetFxPricesData` | `spotfx_prices/FXCODE.parquet` | Single exchange-rate series column `price` |

## Data layout

```text
<parquet_root>/
├── futures_contract_prices/
│   ├── INSTRUMENT#YYYYMM00.parquet          # merged/mixed-frequency contract bars
│   └── Day@INSTRUMENT#YYYYMM00.parquet      # frequency-specific bars, when stored
├── futures_multiple_prices/
│   └── INSTRUMENT.parquet                   # PRICE/FORWARD/CARRY prices and contract ids
├── futures_adjusted_prices/
│   └── INSTRUMENT.parquet                   # stitched continuous price
├── contract_positions/
│   └── INSTRUMENT#YYYYMM00.parquet          # actual held contract position history
├── strategy_positions/
│   └── STRATEGY_INSTRUMENT.parquet          # actual held position attributed to strategy
├── optimal_positions/
│   └── STRATEGY_INSTRUMENT.parquet          # target buffers (lower/upper) from backtest
└── spotfx_prices/
    └── FXCODE.parquet                       # FX spot rates (e.g. EURUSD.parquet)

<backtest_store_directory>/
└── STRATEGY/
    ├── YYYYMMDD_HHMMSS_backtest.pckz        # pickled full backtest state
    └── YYYYMMDD_HHMMSS_config.yaml          # YAML snapshot of config used for backtest

MongoDB production database
├── futures_contracts                        # contract_key = INSTRUMENT/YYYYMM00 metadata & sampling flag
├── INSTRUMENT_ORDER_STACK                   # virtual strategy-level orders per instrument
├── CONTRACT_ORDER_STACK                     # orders allocated to specific contracts
└── BROKER_ORDER_STACK                       # executable broker orders submitted to IB
```

## How backtest results feed order generation

1. **Backtest Execution (`run_systems` / `update_system_backtests`):**
   - Scheduled process runs `runSystemClassic.run_backtest()`.
   - Reads historical adjusted prices, multiple prices, FX rates, and allocated strategy capital.
   - Computes target positions and portfolio buffers (`top_pos`, `bot_pos`) for each instrument.
   - **Writes Optimal Positions:** Writes the latest buffer targets (`lower_position`, `upper_position`, `reference_price`, `reference_contract`) to `optimal_positions/STRATEGY_INSTRUMENT.parquet`.
   - **Saves State Pickle:** Serializes the backtest `System` instance to `<backtest_store_directory>/STRATEGY/<TIMESTAMP>_backtest.pckz`.

2. **Strategy Order Generation (`run_strategy_order_generator` / `update_strategy_orders`):**
   - Scheduled process calls `orderGeneratorForBufferedPositions.get_and_place_orders()`.
   - **Reads Targets:** Fetches target position buffers from `optimal_positions/STRATEGY_INSTRUMENT.parquet`.
   - **Reads Actuals:** Reads current actual strategy position from `strategy_positions/STRATEGY_INSTRUMENT.parquet`.
   - **Buffer Logic:**
     - If `actual < lower_position`: required trade = `round(lower_position) - actual`.
     - If `actual > upper_position`: required trade = `round(upper_position) - actual`.
     - Otherwise (`lower <= actual <= upper`): required trade = 0 (position stays inside buffer band).
   - **Generates Virtual Order:** Constructs an `instrumentOrder` with required trade quantity, reference price, reference contract, and order type (`best_execution`).
   - Applies position limits, locks, and manual overrides.
   - **Pushes to Order Stack:** Submits virtual order to MongoDB `INSTRUMENT_ORDER_STACK`.

3. **Downstream Execution Pipeline:**
   - Stack handler processes `INSTRUMENT_ORDER_STACK` orders, translates virtual instrument trades into specific contract trades (`CONTRACT_ORDER_STACK`), and generates executable `BROKER_ORDER_STACK` items sent to IB.
   - Fills update `contract_positions` and `strategy_positions`, which closes the loop for the next order generation cycle.

## Relationship between market data and position stores

1. Multiple prices define the current `PRICE`, `FORWARD`, and `CARRY` contracts for an instrument.
2. `update_sampled_contracts` reads the furthest-out contract in multiple prices, expands the required contract chain, and upserts those contracts in Mongo with sampling on.
3. `update_historical_prices` downloads prices for contracts whose Mongo metadata has sampling on, and writes per-contract Parquet files.
4. `update_multiple_adjusted_prices` reads per-contract prices plus existing multiple/adjusted prices, then rewrites/appends the multiple and adjusted Parquet files.
5. Production backtests consume adjusted prices, multiple prices, FX prices, and capital data to generate `optimal_positions`.
6. Order generators compare `optimal_positions` against actual `strategy_positions` to emit virtual orders to `INSTRUMENT_ORDER_STACK`.

`traded contracts` are separate from `sampled contracts`: sampled means “download historical prices for this contract”; traded means “the stored actual position series currently ends non-zero”.

## Lifecycle utilities

### `purge_instrument`

`util/purge_instrument.py::purge_instrument(INSTRUMENT)` deletes market-data storage for an instrument:

- Deletes `futures_multiple_prices/INSTRUMENT.parquet` if present.
- Deletes `futures_adjusted_prices/INSTRUMENT.parquet` if present.
- Deletes all per-contract price files for that instrument via `delete_merged_prices_for_instrument_code`.
- Deletes all Mongo `futures_contracts` metadata rows for that instrument.

It does **not** delete FX prices, contract position history, strategy positions, orders, roll state, roll parameters, or instrument config.

### `bootstrap_instrument` / Barchart bootstrap

The bootstrap utility is `util/bootstrap_from_barchart_csv.py::bootstrap_from_barchart_csv`.

Flow:

1. Import Barchart per-contract CSVs into the production contract-price store.
2. Rebuild the roll calendar CSV in `roll_calendar_store` from those contract prices.
3. Build multiple prices from contract prices plus the roll calendar and write `futures_multiple_prices/INSTRUMENT.parquet`.
4. Stitch adjusted prices from multiple prices and write `futures_adjusted_prices/INSTRUMENT.parquet`.
5. Run `update_active_contracts_with_data`, which initializes/refreshes Mongo `futures_contracts` metadata and sampling flags.
6. Optional `update_historical=True`: download current IB historical data, then refresh multiple/adjusted prices again.

Bootstrap creates the storage needed for an instrument to enter the normal daily update cycle.

### `revive_instrument`

`util/revive_instrument.py::revive_instrument(INSTRUMENT, options)` fixes an instrument whose multiple prices point at expired/missing contracts.

Flow:

1. Read current `PRICE` contract from multiple prices.
2. Repeatedly roll with `rollingAdjustedAndMultiplePrices.write_new_rolled_data()` until the priced contract’s desired roll date is after `options.as_of_date`.
3. If stale Mongo metadata is missing, seed minimal `futures_contracts` rows for the current and next roll-path contracts. These rows are not marked sampling by the seeding step.
4. Run `update_active_contracts_with_data` unless skipped. This turns sampling on for the live chain, updates broker expiry dates, and turns sampling off for expired/out-of-chain contracts.
5. Download historical prices unless skipped.
6. Run `update_multiple_adjusted_prices_for_instrument` to refresh multiple and adjusted Parquet files.

Revive does not invent unavailable historical prices; gaps can remain when old contracts are no longer available from IB.
