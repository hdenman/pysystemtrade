import time
import datetime
from syscontrol.run_process import processToRun
from sysexecution.stack_handler.stack_handler import stackHandler
from sysdata.data_blob import dataBlob
from sysbrokers.broker_factory import get_broker_class_list
from syscore.objects import get_class_name

MAX_IB_RETRY_SECONDS = 15 * 60  # 15 minutes
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0


def is_ib_disconnect_exception(exc: BaseException) -> bool:
    if isinstance(exc, ConnectionError):
        return True
    msg = str(exc).lower()
    if "socket disconnect" in msg or "peer closed connection" in msg:
        return True
    return False


def reset_ib_data_blob_and_handler_state(stack_handler_obj: stackHandler):
    data = stack_handler_obj.data
    log = data.log

    try:
        if getattr(data, "_ib_conn", None) is not None and data._ib_conn != getattr(data, "arg_not_supplied", None):
            ib_conn = data._ib_conn
            try:
                ib_conn.close_connection()
            except Exception as e:
                log.warning(f"Error closing old IB connection during reset: {e}")
            try:
                data.db_ib_broker_client_id.release_clientid(ib_conn.client_id())
            except Exception as e:
                log.warning(f"Error releasing client ID during reset: {e}")
    except Exception as e:
        log.warning(f"Error handling ib_conn during reset: {e}")

    data._ib_conn = getattr(data, "arg_not_supplied", None)

    try:
        broker_class_list = get_broker_class_list(data)
        for class_object in broker_class_list:
            class_name = get_class_name(class_object)
            new_name = data._get_new_name(class_name)
            if hasattr(data, new_name):
                try:
                    delattr(data, new_name)
                except Exception:
                    pass
            if hasattr(data, "_attr_list") and new_name in data._attr_list:
                try:
                    data._attr_list.remove(new_name)
                except Exception:
                    pass
    except Exception as e:
        log.warning(f"Error clearing broker attributes during reset: {e}")

    if hasattr(stack_handler_obj, "_data_broker"):
        try:
            delattr(stack_handler_obj, "_data_broker")
        except Exception:
            pass


class RobustIBStackHandler:
    def __init__(
        self,
        stack_handler_obj: stackHandler,
        max_retry_seconds: float = MAX_IB_RETRY_SECONDS,
        time_fn=time.monotonic,
        sleep_fn=time.sleep,
    ):
        self._stack_handler = stack_handler_obj
        self._max_retry_seconds = max_retry_seconds
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn

    @property
    def stack_handler(self) -> stackHandler:
        return self._stack_handler

    @property
    def data(self):
        return self.stack_handler.data

    def __getattr__(self, name: str):
        attr = getattr(self.stack_handler, name)
        if callable(attr):
            def wrapped(*args, **kwargs):
                return self._run_with_ib_retry(name, attr, *args, **kwargs)
            return wrapped
        return attr

    def _run_with_ib_retry(self, method_name: str, method_callable, *args, **kwargs):
        start_time = self._time_fn()
        backoff = INITIAL_BACKOFF_SECONDS
        attempt = 1

        while True:
            try:
                return method_callable(*args, **kwargs)
            except Exception as exc:
                if not is_ib_disconnect_exception(exc):
                    raise

                elapsed = self._time_fn() - start_time
                if elapsed >= self._max_retry_seconds:
                    msg = (
                        f"run_stack_handler method '{method_name}' stuck retrying after "
                        f"IB disconnect for {elapsed:.1f}s (exceeded limit of {self._max_retry_seconds}s). Crashing!"
                    )
                    self.data.log.critical(msg)
                    raise

                next_backoff = min(backoff, MAX_BACKOFF_SECONDS)
                if elapsed + next_backoff > self._max_retry_seconds:
                    next_backoff = max(0.0, self._max_retry_seconds - elapsed)

                self.data.log.warning(
                    f"IB disconnect in run_stack_handler method '{method_name}' (attempt {attempt}, "
                    f"elapsed {elapsed:.1f}s). Reconnecting and retrying in {next_backoff:.1f}s..."
                )

                reset_ib_data_blob_and_handler_state(self.stack_handler)
                self._sleep_fn(next_backoff)

                attempt += 1
                backoff = min(backoff * 2.0, MAX_BACKOFF_SECONDS)


def run_stack_handler():
    process_name = "run_stack_handler"
    data = dataBlob(log_name=process_name)
    list_of_timer_names_and_functions = get_list_of_timer_functions_for_stack_handler()
    price_process = processToRun(process_name, data, list_of_timer_names_and_functions)
    price_process.run_process()


def get_list_of_timer_functions_for_stack_handler():
    stack_handler_data = dataBlob(log_name="stack_handler")
    stack_handler = stackHandler(stack_handler_data)
    stack_handler.set_market_closed_order_backoff(True)
    robust_stack_handler = RobustIBStackHandler(stack_handler)

    list_of_timer_names_and_functions = [
        ("check_external_position_break", robust_stack_handler),
        ("spawn_children_from_new_instrument_orders", robust_stack_handler),
        ("generate_force_roll_orders", robust_stack_handler),
        ("create_broker_orders_from_contract_orders", robust_stack_handler),
        ("process_fills_stack", robust_stack_handler),
        ("handle_completed_orders", robust_stack_handler),
        ("safe_stack_removal", robust_stack_handler),
        ("refresh_additional_sampling_all_instruments", robust_stack_handler),
    ]

    return list_of_timer_names_and_functions


if __name__ == "__main__":
    run_stack_handler()
