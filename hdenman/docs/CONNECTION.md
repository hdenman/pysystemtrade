# Connection resilience

## Problem

`run_stack_handler` is long-lived and depends on IBKR/TWS/Gateway staying usable. IBKR error `1100` means connectivity between IBKR and Trader Workstation/Gateway has been lost. `ib_async` logs that event, but it does not necessarily raise an exception in the caller. Separately, `ib_async.IB.RequestTimeout` defaults to `0`, so blocking requests can wait forever. The failure mode can therefore be a live-but-stuck stack handler rather than a clean crash.

## Approach

Connection resilience is handled in three layers.

## Layer 1: track IBKR connectivity state

`sysbrokers/IB/ib_connection.py` registers an `ib.errorEvent` handler on each `IB()` connection.

- Error `1100` marks the connection unhealthy.
- Errors `1101` and `1102` mark it restored.
- `has_active_connection_problem()` reports unhealthy state when either:
  - the local IB API socket is disconnected; or
  - IBKR/TWS/Gateway connectivity is marked lost by error `1100`.
- `connection_problem_description()` gives the reason used by retry logs/exceptions.

The connection also sets:

```python
IB_REQUEST_TIMEOUT_SECONDS = 15 * 60
ib.RequestTimeout = IB_REQUEST_TIMEOUT_SECONDS
```

This prevents blocking IB requests from hanging indefinitely.

## Layer 2: reconnect and retry stack-handler timer methods

`sysproduction/run_stack_handler.py` wraps the concrete `stackHandler` with `RobustIBStackHandler`.

Before each scheduled method call it checks the current IB connection health. If IB is unhealthy, or the method raises a retryable connection/timeout exception, it:

1. closes the old IB connection if present;
2. releases the old IB client id;
3. clears cached broker data objects from the `dataBlob`;
4. clears cached `stackHandler._data_broker`;
5. sleeps with exponential backoff;
6. retries the same stack-handler method, forcing lazy construction of a fresh IB connection.

Retryable cases include:

- `ConnectionError`
- `TimeoutError`
- retryable socket `OSError` errno values: `ECONNRESET`, `ECONNABORTED`, `EPIPE`, `ETIMEDOUT`
- known IB/ib_async message fragments such as `socket disconnect`, `peer closed connection`, `API connection failed`, `not connected`, and `Connectivity between IBKR and Trader Workstation has been lost`.

The retry window is 15 minutes. If IB is still unavailable after that, the wrapper logs a critical message and skips that timer method run. The process remains alive so the scheduler can try again on the next cycle.

## Layer 3: break active algo loops out of stale state

Some execution algos manage live trades in tight polling loops. If IB reports error `1100` while such a loop is active, no exception may be raised naturally. The loops can keep polling stale control objects.

`sysexecution/algos/common_functions.py` provides `raise_if_active_broker_connection_problem(data)`.

The active management loops in:

- `sysexecution/algos/algo_market.py`
- `sysexecution/algos/algo_original_best.py`
- `sysexecution/algos/common_functions.py` cancel wait loop

call this health check while polling. When IB is unhealthy, it raises `ConnectionError`, which is caught by `RobustIBStackHandler` and converted into the reconnect/retry path above.

## Expected behavior

When IBKR/TWS/Gateway connectivity drops:

1. `ib_async.wrapper` may log error `1100`.
2. The connection object marks itself unhealthy.
3. The next stack-handler method call, or active algo poll loop, raises a retryable connection error.
4. `RobustIBStackHandler` closes and clears the old IB connection state.
5. It retries with exponential backoff for up to 15 minutes.
6. If reconnect succeeds, the interrupted method is retried with fresh broker objects.
7. If reconnect does not succeed within 15 minutes, the method run is skipped and logged critical; `run_stack_handler` remains alive for later scheduled retries.

## Operational note

A live order during an IB outage remains risky. This change avoids an indefinite hang and forces reconnect/reconciliation paths to run, but broker reality can still diverge while the API is unavailable. Existing broker order/fill reconciliation remains responsible for discovering fills and cancellations after connectivity returns.
