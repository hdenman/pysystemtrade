# Plan: split pysystemtrade into `pst-paper` and `pst-live` Unix users

## Goal

Run two active pysystemtrade production systems on one machine:

- `pst-paper`: existing system migrated from the current `hdenman` setup; trades IB paper account.
- `pst-live`: new clean live system; trades IB live account.
- Historic market data is collected by `pst-live` and consumed by `pst-paper`.
- Execution/accounting/process state stays separate.

Core rule:

> Share market data; do not share trading state.

Market data means:

- `futures_contract_prices`
- `futures_multiple_prices`
- `futures_adjusted_prices`
- `spotfx_prices`
- optionally `spreads`, if you treat spread histories as shared market data

Trading state means:

- order stacks
- broker/order history
- capital
- current/historic positions
- optimal positions
- roll state
- process control
- logs/echos/backtest state

## Target layout

Recommended filesystem layout:

```text
/srv/pysystemtrade/
  shared-parquet/
    futures_contract_prices/
    futures_multiple_prices/
    futures_adjusted_prices/
    spotfx_prices/
    spreads/                         # optional shared market data

  paper/
    pysystemtrade/                    # code checkout
    pysystemtrade_config/             # private config repo/copy
    parquet/
      futures_contract_prices -> ../../shared-parquet/futures_contract_prices
      futures_multiple_prices -> ../../shared-parquet/futures_multiple_prices
      futures_adjusted_prices -> ../../shared-parquet/futures_adjusted_prices
      spotfx_prices -> ../../shared-parquet/spotfx_prices
      spreads -> ../../shared-parquet/spreads
      capital/
      contract_positions/
      strategy_positions/
      optimal_positions/
    echoes/
    logs/
    backtest_states/
    ibgateway/                       # paper Gateway/IBC runtime/config

  live/
    pysystemtrade/                    # code checkout
    pysystemtrade_config/             # private config repo/copy
    parquet/
      futures_contract_prices -> ../../shared-parquet/futures_contract_prices
      futures_multiple_prices -> ../../shared-parquet/futures_multiple_prices
      futures_adjusted_prices -> ../../shared-parquet/futures_adjusted_prices
      spotfx_prices -> ../../shared-parquet/spotfx_prices
      spreads -> ../../shared-parquet/spreads
      capital/
      contract_positions/
      strategy_positions/
      optimal_positions/
    echoes/
    logs/
    backtest_states/
    ibgateway/                       # live Gateway/IBC runtime/config
```

Alternative: make `pst-live` write directly to `/srv/pysystemtrade/live/parquet` and symlink only the paper market-data directories to live. The explicit `shared-parquet` directory is clearer because it separates shared data from live trading state.

## Users and groups

Create users:

```bash
sudo useradd --system --create-home --home-dir /srv/pysystemtrade/paper --shell /bin/bash pst-paper
sudo useradd --system --create-home --home-dir /srv/pysystemtrade/live --shell /bin/bash pst-live
```

Create a shared group for read access to market data:

```bash
sudo groupadd pst-data
sudo usermod -aG pst-data pst-paper
sudo usermod -aG pst-data pst-live
```

Set ownership policy:

```bash
sudo mkdir -p /srv/pysystemtrade/shared-parquet
sudo chown -R pst-live:pst-data /srv/pysystemtrade/shared-parquet
sudo chmod -R 2775 /srv/pysystemtrade/shared-parquet
```

If you want stronger safety, make `pst-paper` read-only on shared market data after migration:

```bash
sudo setfacl -R -m u:pst-live:rwx,u:pst-paper:rx,g:pst-data:rx /srv/pysystemtrade/shared-parquet
sudo setfacl -R -d -m u:pst-live:rwx,u:pst-paper:rx,g:pst-data:rx /srv/pysystemtrade/shared-parquet
```

On NixOS, prefer declarative users/groups in `/etc/nixos/configuration.nix` or your flake module. Equivalent shape:

```nix
users.groups.pst-data = {};

users.users.pst-paper = {
  isSystemUser = true;
  home = "/srv/pysystemtrade/paper";
  createHome = true;
  group = "pst-paper";
  extraGroups = [ "pst-data" ];
  shell = pkgs.bashInteractive;
};

users.users.pst-live = {
  isSystemUser = true;
  home = "/srv/pysystemtrade/live";
  createHome = true;
  group = "pst-live";
  extraGroups = [ "pst-data" ];
  shell = pkgs.bashInteractive;
};
```

Then apply with:

```bash
sudo nixos-rebuild switch
```

## Phase 1: migrate the existing `hdenman` system to `pst-paper`

### 1. Freeze current production jobs

Stop cron/systemd jobs for the existing `hdenman` setup before copying state.

Checklist:

- Stop pysystemtrade cron entries.
- Stop existing IB Gateway systemd unit.
- Stop long-running stack handler if managed separately.
- Confirm no update/order process is running.

Typical commands, adjusted to your actual unit names:

```bash
sudo systemctl stop ibgateway.service
sudo systemctl stop pysystemtrade-stack-handler.service  # only if you have one
sudo crontab -u hdenman -l > /home/hdenman/pysystemtrade.crontab.backup
sudo crontab -u hdenman -r
```

If using the supplied crontab only, preserve it first and remove it only after the new paper crontab is installed.

### 2. Backup before moving anything

Create cold backups of code, config, Mongo, and Parquet.

```bash
sudo mkdir -p /srv/pysystemtrade/backups/pre-split
sudo rsync -aH --numeric-ids /home/hdenman/algo-trading/pysystemtrade/ /srv/pysystemtrade/backups/pre-split/pysystemtrade/
sudo rsync -aH --numeric-ids /home/hdenman/algo-trading/pysystemtrade_config/ /srv/pysystemtrade/backups/pre-split/pysystemtrade_config/
sudo rsync -aH --numeric-ids /home/hdenman/algo-trading/pysystemtrade-data/ /srv/pysystemtrade/backups/pre-split/pysystemtrade-data/
```

Back up Mongo with `mongodump` if Mongo is running:

```bash
sudo -u hdenman mongodump --out /srv/pysystemtrade/backups/pre-split/mongodump
```

Do not delete the original `hdenman` tree until both paper and live have run successfully for several days.

### 3. Copy existing code and private config to `pst-paper`

```bash
sudo mkdir -p /srv/pysystemtrade/paper
sudo rsync -aH /home/hdenman/algo-trading/pysystemtrade/ /srv/pysystemtrade/paper/pysystemtrade/
sudo rsync -aH /home/hdenman/algo-trading/pysystemtrade_config/ /srv/pysystemtrade/paper/pysystemtrade_config/
sudo chown -R pst-paper:pst-paper /srv/pysystemtrade/paper
```

If the current system uses symlinks in `private/`, recreate them under the new user rather than copying absolute links blindly.

Recommended convention:

```text
/srv/pysystemtrade/paper/pysystemtrade/private -> /srv/pysystemtrade/paper/pysystemtrade_config/private
```

Create or fix the link:

```bash
sudo -u pst-paper ln -sfn /srv/pysystemtrade/paper/pysystemtrade_config/private /srv/pysystemtrade/paper/pysystemtrade/private
```

### 4. Move/copy current Parquet data

If the current system is the paper system, copy its Parquet store into paper first:

```bash
sudo mkdir -p /srv/pysystemtrade/paper/parquet
sudo rsync -aH /home/hdenman/algo-trading/pysystemtrade-data/parquet/ /srv/pysystemtrade/paper/parquet/
sudo chown -R pst-paper:pst-paper /srv/pysystemtrade/paper/parquet
```

Then carve out shared market-data directories:

```bash
sudo mkdir -p /srv/pysystemtrade/shared-parquet

for d in futures_contract_prices futures_multiple_prices futures_adjusted_prices spotfx_prices spreads; do
  if [ -e /srv/pysystemtrade/paper/parquet/$d ] && [ ! -e /srv/pysystemtrade/shared-parquet/$d ]; then
    sudo mv /srv/pysystemtrade/paper/parquet/$d /srv/pysystemtrade/shared-parquet/$d
  fi
  sudo -u pst-paper ln -sfn ../../shared-parquet/$d /srv/pysystemtrade/paper/parquet/$d
done

sudo chown -R pst-live:pst-data /srv/pysystemtrade/shared-parquet
sudo chmod -R 2775 /srv/pysystemtrade/shared-parquet
```

Rationale: paper starts by consuming the same market data it had before, but ownership of future shared-data writes is transferred to `pst-live`.

### 5. Migrate Mongo state for paper

Set paper Mongo DB name to something explicit:

```yaml
mongo_db: production_paper
```

If existing DB was named `production`, copy it to `production_paper` before starting paper. Exact DB names may be suffixed by your universe setting in code, because `mongoDb` appends `_<universe_subdir()>`.

Procedure:

1. Identify actual existing database name in Mongo.
2. Dump it.
3. Restore under the new paper DB name.

Example:

```bash
mongodump --db production --out /srv/pysystemtrade/backups/pre-split/mongo-production
mongorestore --db production_paper /srv/pysystemtrade/backups/pre-split/mongo-production/production
```

If the actual DB is `production_<universe>`, use that exact source and restore to `production_paper_<universe>`.

Paper must keep its own Mongo DB because Mongo stores order stacks, historic orders, roll state, process control, and contract metadata.

### 6. Configure `pst-paper` private config

In `pst-paper` private config:

```yaml
mongo_host: 127.0.0.1
mongo_db: production_paper
parquet_store: /srv/pysystemtrade/paper/parquet

ib_ipaddress: 127.0.0.1
ib_port: 4002
ib_idoffset: 1000
broker_account: DU123456       # replace with actual paper account
```

Use a high `ib_idoffset` for paper to avoid client ID collisions with live.

Keep paper logs/echos/backtest state under paper-owned directories. Ensure `~pst-paper/.profile` exports paths for the paper instance:

```bash
export PYSYSTEMTRADE_HOME=/srv/pysystemtrade/paper/pysystemtrade
export SCRIPT_PATH=/srv/pysystemtrade/paper/pysystemtrade/sysproduction/linux/scripts
export ECHO_PATH=/srv/pysystemtrade/paper/echoes
export MONGO_DATA=/srv/pysystemtrade/mongo-data
export PARQUET_DATA=/srv/pysystemtrade/paper/parquet
cd "$PYSYSTEMTRADE_HOME"
```

If your setup uses different variables, keep those; the invariant is that `pst-paper` must resolve config, scripts, echos, and Parquet to the paper tree.

### 7. Configure paper IB Gateway systemd unit

Paper needs its own IB Gateway instance if it will execute paper trades or poll paper account positions/capital/orders.

Use a distinct service, runtime directory, config directory, API port, and login mode.

Example unit shape:

```ini
[Unit]
Description=IB Gateway paper
After=network-online.target
Wants=network-online.target

[Service]
User=pst-paper
Group=pst-paper
WorkingDirectory=/srv/pysystemtrade/paper/ibgateway
Environment=IBC_INI=/srv/pysystemtrade/paper/ibgateway/ibc-paper.ini
Environment=TWS_SETTINGS_PATH=/srv/pysystemtrade/paper/ibgateway/tws-settings
ExecStart=/srv/ibc/scripts/ibcstart.sh gateway paper
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Exact `ExecStart` depends on your current IBC/Gateway install. The required properties are:

- Paper login/session, not live.
- API socket port matches paper config, e.g. `4002`.
- API allows connections from `127.0.0.1`.
- Read-only API is off if paper trading will place orders.
- Runtime/settings directory is not shared with live.

Enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ibgateway-paper.service
```

### 8. Install initial paper cron

At this point, install paper cron without shared market-data writers. See Phase 3 for the final paper crontab.

## Phase 2: set up new `pst-live` user

### 1. Create live directories

```bash
sudo mkdir -p /srv/pysystemtrade/live/{pysystemtrade,pysystemtrade_config,parquet,echoes,logs,backtest_states,ibgateway}
sudo chown -R pst-live:pst-live /srv/pysystemtrade/live
```

### 2. Install code for live

Use the same commit as paper initially. Do not let live and paper drift during the migration.

```bash
sudo rsync -aH /srv/pysystemtrade/paper/pysystemtrade/ /srv/pysystemtrade/live/pysystemtrade/
sudo rsync -aH /srv/pysystemtrade/paper/pysystemtrade_config/ /srv/pysystemtrade/live/pysystemtrade_config/
sudo chown -R pst-live:pst-live /srv/pysystemtrade/live
sudo -u pst-live ln -sfn /srv/pysystemtrade/live/pysystemtrade_config/private /srv/pysystemtrade/live/pysystemtrade/private
```

Then edit live private config separately.

### 3. Create live Parquet layout

Live should write shared market-data directories, while keeping live trading-state directories local.

```bash
sudo mkdir -p /srv/pysystemtrade/live/parquet

for d in futures_contract_prices futures_multiple_prices futures_adjusted_prices spotfx_prices spreads; do
  sudo -u pst-live ln -sfn ../../shared-parquet/$d /srv/pysystemtrade/live/parquet/$d
done

sudo -u pst-live mkdir -p \
  /srv/pysystemtrade/live/parquet/capital \
  /srv/pysystemtrade/live/parquet/contract_positions \
  /srv/pysystemtrade/live/parquet/strategy_positions \
  /srv/pysystemtrade/live/parquet/optimal_positions
```

### 4. Configure `pst-live` private config

Live config:

```yaml
mongo_host: 127.0.0.1
mongo_db: production_live
parquet_store: /srv/pysystemtrade/live/parquet

ib_ipaddress: 127.0.0.1
ib_port: 4001
ib_idoffset: 1
broker_account: U123456        # replace with actual live account
```

Do not reuse the paper `broker_account`. Do not rely on IB default account selection.

### 5. Initialise live Mongo DB

Live should not inherit paper order stacks, broker history, capital, position, or process-control state.

Options:

- Clean start: create empty `production_live` DB and initialise capital/process state through normal pysystemtrade setup.
- Selective copy: copy only static-like collections if you know exactly which ones are safe.

Boring/safe recommendation: clean start. Contract metadata can be generated by `update_sampled_contracts` in live.

### 6. Configure live IB Gateway systemd unit

Live requires its own Gateway/IBC instance if live will trade.

Example unit shape:

```ini
[Unit]
Description=IB Gateway live
After=network-online.target
Wants=network-online.target

[Service]
User=pst-live
Group=pst-live
WorkingDirectory=/srv/pysystemtrade/live/ibgateway
Environment=IBC_INI=/srv/pysystemtrade/live/ibgateway/ibc-live.ini
Environment=TWS_SETTINGS_PATH=/srv/pysystemtrade/live/ibgateway/tws-settings
ExecStart=/srv/ibc/scripts/ibcstart.sh gateway live
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Required properties:

- Live login/session.
- API socket port matches live config, normally `4001`.
- API allows `127.0.0.1`.
- Read-only API is off only when you are ready for live order placement.
- Live Gateway settings directory is separate from paper.

Enable it when ready:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ibgateway-live.service
```

### 7. Install live cron

Live owns shared data collection. It should run the price/data jobs.

Example live crontab:

```cron
# pst-live crontab

# Long-running stack handler / trading loop
15 00 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_stack_handler >> $ECHO_PATH/run_stack_handler.txt 2>&1

# Live account capital and allocation
45 00 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_capital_update >> $ECHO_PATH/run_capital_update.txt 2>&1

# Shared market-data writes: live only
30 06 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_daily_fx_and_contract_updates >> $ECHO_PATH/run_daily_fx_and_contract_updates.txt 2>&1
05 07 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_daily_price_updates >> $ECHO_PATH/run_daily_price_updates.txt 2>&1
00 19 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_daily_update_multiple_adjusted_prices >> $ECHO_PATH/run_daily_update_multiple_adjusted_prices.txt 2>&1

# Strategy/order flow after shared data is updated
30 20 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_systems >> $ECHO_PATH/run_systems.txt 2>&1
45 20 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_strategy_order_generator >> $ECHO_PATH/run_strategy_order_generator.txt 2>&1

# Housekeeping
00 21 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_cleaners >> $ECHO_PATH/run_cleaners.txt 2>&1
15 21 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_backups >> $ECHO_PATH/run_backups.txt 2>&1
30 21 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_reports >> $ECHO_PATH/run_reports.txt 2>&1

# Startup
@reboot $HOME/.profile; $SCRIPT_PATH/startup >> $ECHO_PATH/startup.txt 2>&1
```

Install:

```bash
sudo -u pst-live crontab /srv/pysystemtrade/live/pysystemtrade/sysproduction/linux/crontab.live
```

## Phase 3: disable data collection in `pst-paper` so it depends on `pst-live`

### 1. Disable shared price writers in paper cron

Paper must not run jobs that write to shared market-data Parquet directories.

Do not schedule these in `pst-paper`:

```cron
# DO NOT RUN IN pst-paper WHEN USING SHARED DATA
# $SCRIPT_PATH/run_daily_price_updates
# $SCRIPT_PATH/run_daily_update_multiple_adjusted_prices
```

Also do not schedule the stock combined script unless you are comfortable with its mixed behavior:

```cron
# CAUTION: mixed shared/local behavior
# $SCRIPT_PATH/run_daily_fx_and_contract_updates
```

Reason: in this repo:

- `run_daily_price_updates` calls `update_historical_prices`, which writes futures contract prices.
- `run_daily_update_multiple_adjusted_prices` calls `update_multiple_adjusted_prices`, which writes multiple and adjusted prices.
- `run_daily_fx_and_contract_updates` calls both:
  - `update_fx_prices`, which writes shared FX prices.
  - `update_sampled_contracts`, which writes contract metadata to Mongo.

### 2. Handle `update_sampled_contracts` explicitly

`update_sampled_contracts` is the special case.

It writes active contract metadata to MongoDB, not to Parquet. If `pst-paper` and `pst-live` use separate Mongo DB names, then running `update_sampled_contracts` in paper updates only paper's local Mongo state.

That can be useful because order generation/execution and diagnostics may need current contract metadata around rolls and expiries.

But the supplied `run_daily_fx_and_contract_updates` is not suitable for paper-as-consumer because it also runs `update_fx_prices`, a shared-data writer.

Recommended solution: add a paper-only wrapper that runs only sampled-contract updates.

Create a local script/module in the paper private/code area, for example:

```python
# /srv/pysystemtrade/paper/pysystemtrade/private/run_daily_sampled_contracts_only.py

from syscontrol.run_process import processToRun
from sysproduction.update_sampled_contracts import updateSampledContracts
from sysdata.data_blob import dataBlob


def run_daily_sampled_contracts_only():
    process_name = "run_daily_sampled_contracts_only"
    data = dataBlob(log_name=process_name)
    updater = updateSampledContracts(data)
    process = processToRun(
        process_name,
        data,
        [("update_sampled_contracts", updater)],
    )
    process.run_process()


if __name__ == "__main__":
    run_daily_sampled_contracts_only()
```

Create a shell wrapper:

```bash
# /srv/pysystemtrade/paper/pysystemtrade/sysproduction/linux/scripts/run_daily_sampled_contracts_only
#!/bin/bash
. ~/.profile
. p private.run_daily_sampled_contracts_only.run_daily_sampled_contracts_only
```

Make it executable:

```bash
sudo chmod +x /srv/pysystemtrade/paper/pysystemtrade/sysproduction/linux/scripts/run_daily_sampled_contracts_only
sudo chown pst-paper:pst-paper /srv/pysystemtrade/paper/pysystemtrade/sysproduction/linux/scripts/run_daily_sampled_contracts_only
```

Paper can then run this safely, because it updates paper Mongo only:

```cron
30 07 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_daily_sampled_contracts_only >> $ECHO_PATH/run_daily_sampled_contracts_only.txt 2>&1
```

If paper does not need fresh local contract metadata, you can skip this wrapper initially. Risk: paper may have stale contract metadata around rolls/expiries.

### 3. Schedule paper after live data completion

Paper should run systems only after live has completed:

1. `run_daily_fx_and_contract_updates`
2. `run_daily_price_updates`
3. `run_daily_update_multiple_adjusted_prices`

Simple cron-based schedule:

```cron
# pst-paper crontab

# Paper trading loop
15 00 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_stack_handler >> $ECHO_PATH/run_stack_handler.txt 2>&1

# Paper account capital/allocation
45 00 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_capital_update >> $ECHO_PATH/run_capital_update.txt 2>&1

# Optional local Mongo-only contract metadata update
30 07 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_daily_sampled_contracts_only >> $ECHO_PATH/run_daily_sampled_contracts_only.txt 2>&1

# No shared price writers here.
# No run_daily_price_updates.
# No run_daily_update_multiple_adjusted_prices.
# No stock run_daily_fx_and_contract_updates unless intentionally modified.

# Strategy/order flow after live has refreshed shared data
35 20 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_systems >> $ECHO_PATH/run_systems.txt 2>&1
50 20 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_strategy_order_generator >> $ECHO_PATH/run_strategy_order_generator.txt 2>&1

# Housekeeping
05 21 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_cleaners >> $ECHO_PATH/run_cleaners.txt 2>&1
20 21 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_backups >> $ECHO_PATH/run_backups.txt 2>&1
35 21 * * 1-5 $HOME/.profile; $SCRIPT_PATH/run_reports >> $ECHO_PATH/run_reports.txt 2>&1

# Startup
@reboot $HOME/.profile; $SCRIPT_PATH/startup >> $ECHO_PATH/startup.txt 2>&1
```

This is simple but time-based. If live data updates run late, paper can still consume stale data.

### 4. Prefer a completion sentinel for stronger dependency

Better: have live write a date-stamped sentinel after successful shared data update, and have paper check it before `run_systems`.

Concept:

```text
pst-live finishes update_multiple_adjusted_prices
  -> writes /srv/pysystemtrade/shared-parquet/.prices-ready/YYYY-MM-DD
pst-paper run_systems wrapper
  -> checks today's sentinel exists
  -> exits non-zero or waits if missing
```

Example live post-update command:

```bash
mkdir -p /srv/pysystemtrade/shared-parquet/.prices-ready
date +%F > /srv/pysystemtrade/shared-parquet/.prices-ready/$(date +%F)
```

Example paper wrapper check:

```bash
READY_FILE="/srv/pysystemtrade/shared-parquet/.prices-ready/$(date +%F)"
if [ ! -f "$READY_FILE" ]; then
  echo "Shared prices not ready: $READY_FILE missing"
  exit 1
fi
. ~/.profile
. p sysproduction.run_systems.run_systems
```

This avoids relying only on clock times.

### 5. Permissions safety

After live owns shared market-data writes, make paper unable to write those directories if practical.

Target:

```text
pst-live: read/write shared market data
pst-paper: read-only shared market data
```

This catches accidental paper scheduling of price updates quickly.

## IB Gateway setup details

### Required instances

Use two Gateway/IBC instances if both systems actively connect to IB:

| System | Unix user | IB mode | API port | Account config | id offset |
|---|---|---|---:|---|---:|
| paper | `pst-paper` | paper | `4002` | `DU...` | `1000` |
| live | `pst-live` | live | `4001` | `U...` | `1` |

Do not share a Gateway process between paper and live.

Reasons:

- A Gateway session is tied to one login/mode.
- Paper and live should have separate API ports.
- Runtime settings and restarts should not collide.
- Operational mistakes in paper should not disturb live.

### Gateway settings per instance

For each Gateway/TWS profile:

- Enable API socket clients.
- Set socket port:
  - live: `4001`
  - paper: `4002`
- Trusted IPs include `127.0.0.1`.
- Disable read-only API only for systems that should place orders.
- Review order precautions/presets separately for live and paper.
- Use distinct IBC config files and TWS/Gateway settings directories.

### pysystemtrade config per instance

Paper:

```yaml
ib_ipaddress: 127.0.0.1
ib_port: 4002
ib_idoffset: 1000
broker_account: DU123456
```

Live:

```yaml
ib_ipaddress: 127.0.0.1
ib_port: 4001
ib_idoffset: 1
broker_account: U123456
```

`ib_idoffset` matters because pysystemtrade allocates IB client IDs. Duplicate client IDs on the same Gateway cause conflicts.

## MongoDB separation

Use separate DB names:

```yaml
# paper
mongo_db: production_paper

# live
mongo_db: production_live
```

Do not share the whole Mongo DB.

Mongo-backed state includes order stacks, historic orders, roll state, process control, futures contract metadata, and stored spread costs. Some of this may look reusable, but sharing the DB mixes live and paper state.

If you later want to share only futures contract metadata, do that as an explicit code/data-object customization. Do not get it accidentally by using the same `mongo_db`.

## Validation checklist

Run these before enabling live trading:

### Paper validation

- `pst-paper` can import/run pysystemtrade from its own checkout.
- `pst-paper` reads adjusted prices from shared market-data symlink.
- `pst-paper` writes capital/positions/optimal positions only under `/srv/pysystemtrade/paper/parquet`.
- `pst-paper` Mongo DB is `production_paper...`, not live.
- Paper Gateway is reachable on `127.0.0.1:4002`.
- Paper `broker_account` is `DU...`.
- Paper crontab has no shared price writer jobs.

### Live validation

- `pst-live` reads/writes shared market-data directories.
- `pst-live` writes capital/positions/optimal positions only under `/srv/pysystemtrade/live/parquet`.
- `pst-live` Mongo DB is `production_live...`, not paper.
- Live Gateway is reachable on `127.0.0.1:4001`.
- Live `broker_account` is `U...`.
- Live crontab owns data collection jobs.

### Cross-system validation

- Paper `run_systems` starts after live shared data update.
- Paper cannot write shared market-data directories if using read-only ACLs.
- Live and paper `ib_idoffset` ranges do not overlap.
- Live and paper systemd Gateway units have separate users, ports, IBC configs, and settings directories.
- Backups cover both Mongo DBs, both local Parquet trees, and the shared market-data tree.

## Cutover order

1. Back up current `hdenman` setup.
2. Create `pst-paper` and `pst-live` users/groups.
3. Copy existing system to `pst-paper`.
4. Configure `pst-paper` as paper account with separate Mongo DB and paper Gateway.
5. Move/symlink market-data directories into shared Parquet.
6. Start and validate `pst-paper` without live changes.
7. Copy code/config skeleton to `pst-live`.
8. Configure `pst-live` as live account with separate Mongo DB and live Gateway.
9. Enable live data collection only after confirming it writes shared market data.
10. Disable paper shared-data collection jobs.
11. Add optional paper `update_sampled_contracts`-only wrapper.
12. Enable paper trading jobs after live data jobs complete.
13. Keep original `hdenman` setup read-only as rollback until stable.

## Rollback plan

If migration fails before live is enabled:

1. Stop `pst-paper` cron and Gateway.
2. Restore `hdenman` crontab from backup.
3. Restart original Gateway unit.
4. Point original config back to original Mongo/Parquet paths if changed.

If live has been enabled:

1. Stop live order generation and stack handler first.
2. Leave data collection running only if it is known good.
3. Stop paper if shared data integrity is in doubt.
4. Restore shared Parquet from pre-split backup if needed.
5. Restore Mongo DBs independently; never restore paper DB over live DB.
