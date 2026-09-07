import datetime
from sysexecution.stack_handler.create_broker_orders_from_contract_orders import (
    stackHandlerCreateBrokerOrders,
    MAX_MARKET_CLOSED_BACKOFF,
)
import sysexecution.stack_handler.create_broker_orders_from_contract_orders as broker_order_module
from sysexecution.orders.contract_orders import contractOrder
from sysexecution.orders.named_order_objects import missing_order
from sysobjects.production.trading_hours.trading_hours import (
    listOfTradingHours,
    tradingHours,
)


class FakeLog:
    def __init__(self):
        self.debug_calls = []
        self.warning_calls = []

    def debug(self, msg, *args, **kwargs):
        self.debug_calls.append((msg % args if args else msg, kwargs))

    def warning(self, msg, *args, **kwargs):
        self.warning_calls.append((msg % args if args else msg, kwargs))


class FakeDataBroker:
    def __init__(self, trading_hours=None, raise_on_get_trading_hours=False):
        self.trading_hours = trading_hours
        self.get_trading_hours_call_count = 0
        self.is_contract_okay_call_count = 0
        self.raise_on_get_trading_hours = raise_on_get_trading_hours

    def get_trading_hours_for_contract(self, contract):
        self.get_trading_hours_call_count += 1
        if self.raise_on_get_trading_hours:
            raise Exception("Trading hours unavailable")
        return self.trading_hours

    def is_contract_okay_to_trade(self, contract):
        self.is_contract_okay_call_count += 1
        if self.trading_hours is not None:
            return self.trading_hours.okay_to_trade_now()
        return True


def make_test_contract_order():
    return contractOrder(
        "test_strat",
        "EDOLLAR",
        "202609",
        1,
    )


def make_handler(monkeypatch, trading_hours=None, raise_on_get_trading_hours=False):
    handler = object.__new__(stackHandlerCreateBrokerOrders)
    handler._data = object()
    handler._log = FakeLog()
    handler._data_broker = FakeDataBroker(
        trading_hours=trading_hours,
        raise_on_get_trading_hours=raise_on_get_trading_hours,
    )

    class FakeDataLocks:
        def __init__(self, data):
            pass

        def is_instrument_locked(self, instrument_code):
            return False

    monkeypatch.setattr(broker_order_module, "dataLocks", FakeDataLocks)
    handler.size_contract_order = lambda original_contract_order: original_contract_order
    return handler


def test_loop_backoff_skips_closed_contract_until_next_opening(monkeypatch):
    now = datetime.datetime.now()
    next_open = now + datetime.timedelta(minutes=20)
    next_close = next_open + datetime.timedelta(hours=5)

    th = listOfTradingHours([tradingHours(next_open, next_close)])

    handler = make_handler(monkeypatch, trading_hours=th)
    handler.set_market_closed_order_backoff(True)
    order = make_test_contract_order()

    # First pass: detects closed market, sets backoff, returns missing_order
    res1 = handler.preprocess_contract_order(order)
    assert res1 is missing_order
    assert handler._data_broker.get_trading_hours_call_count == 1

    contract_key = order.futures_contract.key
    stored_retry = handler.market_closed_contract_order_backoff.get(contract_key)
    assert stored_retry is not None
    assert stored_retry == next_open

    # Second pass before retry time: skipped by backoff without checking trading hours again
    res2 = handler.preprocess_contract_order(order)
    assert res2 is missing_order
    assert handler._data_broker.get_trading_hours_call_count == 1
    # Backoff skips evaluation silently without logging
    assert not any("backoff in effect" in msg for msg, _ in handler._log.debug_calls)


def test_loop_backoff_is_capped_at_one_hour(monkeypatch):
    now = datetime.datetime.now()
    next_open = now + datetime.timedelta(hours=5)
    next_close = next_open + datetime.timedelta(hours=5)

    th = listOfTradingHours([tradingHours(next_open, next_close)])

    handler = make_handler(monkeypatch, trading_hours=th)
    handler.set_market_closed_order_backoff(True)
    order = make_test_contract_order()

    start_time = datetime.datetime.now()
    res = handler.preprocess_contract_order(order)
    assert res is missing_order

    contract_key = order.futures_contract.key
    stored_retry = handler.market_closed_contract_order_backoff.get(contract_key)
    assert stored_retry is not None
    assert stored_retry >= start_time + datetime.timedelta(minutes=55)
    assert stored_retry <= start_time + datetime.timedelta(hours=1, seconds=5)


def test_expired_loop_backoff_rechecks_and_clears_when_market_open(monkeypatch):
    now = datetime.datetime.now()
    open_start = now - datetime.timedelta(hours=1)
    open_end = now + datetime.timedelta(hours=1)

    th = listOfTradingHours([tradingHours(open_start, open_end)])

    handler = make_handler(monkeypatch, trading_hours=th)
    handler.set_market_closed_order_backoff(True)
    order = make_test_contract_order()

    # Pre-populate backoff entry with an expired datetime
    contract_key = order.futures_contract.key
    handler.market_closed_contract_order_backoff[contract_key] = now - datetime.timedelta(minutes=1)

    res = handler.preprocess_contract_order(order)
    assert res is order
    assert contract_key not in handler.market_closed_contract_order_backoff


def test_manual_direct_path_ignores_existing_backoff_and_logs_warning(monkeypatch):
    now = datetime.datetime.now()
    next_open = now + datetime.timedelta(hours=2)
    next_close = next_open + datetime.timedelta(hours=5)

    th = listOfTradingHours([tradingHours(next_open, next_close)])

    handler = make_handler(monkeypatch, trading_hours=th)
    # Default (use_market_closed_order_backoff == False)
    assert handler.use_market_closed_order_backoff is False

    order = make_test_contract_order()
    contract_key = order.futures_contract.key
    handler.market_closed_contract_order_backoff[contract_key] = now + datetime.timedelta(minutes=30)

    res = handler.preprocess_contract_order(order)
    assert res is missing_order
    # Checked trading hours despite existing backoff entry
    assert handler._data_broker.get_trading_hours_call_count == 1
    # Logged warning, not debug
    assert len(handler._log.warning_calls) == 1
    assert "Order" in handler._log.warning_calls[0][0]
