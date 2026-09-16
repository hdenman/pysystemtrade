import errno
import time
from typing import Any, Iterator
from sysbrokers.broker_factory import get_broker_class_list
from syscore.objects import get_class_name
from syscore.constants import arg_not_supplied

MAX_IB_RETRY_SECONDS = 15 * 60  # 15 minutes
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0

IB_RETRYABLE_ERRNOS = {
    errno.ECONNRESET,
    errno.ECONNABORTED,
    errno.EPIPE,
    errno.ETIMEDOUT,
}

IB_RETRYABLE_MESSAGE_FRAGMENTS = (
    "socket disconnect",
    "peer closed connection",
    "api connection failed",
    "connection timed out",
    "timed out",
    "timeouterror",
    "not connected",
    "connectivity between ibkr and trader workstation has been lost",
)


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen = set()
    curr = exc
    while curr is not None and id(curr) not in seen:
        seen.add(id(curr))
        yield curr
        if curr.__cause__ is not None and id(curr.__cause__) not in seen:
            curr = curr.__cause__
        elif curr.__context__ is not None and id(curr.__context__) not in seen:
            curr = curr.__context__
        else:
            break


def is_ib_disconnect_exception(exc: BaseException) -> bool:
    for chained_exc in _iter_exception_chain(exc):
        if isinstance(chained_exc, (ConnectionError, TimeoutError)):
            return True
        if isinstance(chained_exc, OSError) and chained_exc.errno in IB_RETRYABLE_ERRNOS:
            return True
        msg = str(chained_exc).lower()
        if any(frag in msg for frag in IB_RETRYABLE_MESSAGE_FRAGMENTS):
            return True
    return False


def _get_existing_ib_connection(data):
    ib_conn = getattr(data, "_ib_conn", arg_not_supplied)
    if ib_conn in (None, arg_not_supplied):
        return arg_not_supplied
    return ib_conn


def raise_if_known_ib_connection_problem(data):
    ib_conn = _get_existing_ib_connection(data)
    if ib_conn is arg_not_supplied:
        return None

    has_active_connection_problem = getattr(
        ib_conn, "has_active_connection_problem", None
    )
    if has_active_connection_problem is None or has_active_connection_problem() is not True:
        return None

    connection_problem_description = getattr(
        ib_conn, "connection_problem_description", None
    )
    if connection_problem_description is None:
        raise ConnectionError("IB connection is not healthy")

    raise ConnectionError(connection_problem_description())


def reset_ib_data_blob_and_target_state(target_obj: Any) -> None:
    data = target_obj.data
    log = data.log
    try:
        ib_conn = _get_existing_ib_connection(data)
        if ib_conn is not arg_not_supplied:
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

    data._ib_conn = arg_not_supplied

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

    if hasattr(target_obj, "_data_broker"):
        try:
            delattr(target_obj, "_data_broker")
        except Exception:
            pass


class RobustIBRetryWrapper:
    def __init__(
        self,
        target_obj: Any,
        process_name: str,
        max_retry_seconds: float = MAX_IB_RETRY_SECONDS,
        time_fn=time.monotonic,
        sleep_fn=time.sleep,
    ):
        self._target_obj = target_obj
        self._process_name = process_name
        self._max_retry_seconds = max_retry_seconds
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn

    @property
    def target_obj( self) -> Any:
        return self._target_obj

    @property
    def data(self):
        return self.target_obj.data

    def __getattr__(self, name: str):
        attr = getattr(self.target_obj, name)
        if callable(attr):
            def wrapped(*args, **kwargs):
                return self._run_with_ib_retry(name, *args, **kwargs)
            return wrapped
        return attr

    def _run_with_ib_retry(self, method_name: str, *args, **kwargs):
        start_time = self._time_fn()
        backoff = INITIAL_BACKOFF_SECONDS
        attempt = 1

        while True:
            try:
                raise_if_known_ib_connection_problem(self.data)
                method_callable = getattr(self.target_obj, method_name)
                return method_callable(*args, **kwargs)
            except Exception as exc:
                if not is_ib_disconnect_exception(exc):
                    raise

                elapsed = self._time_fn() - start_time
                if elapsed >= self._max_retry_seconds:
                    msg = (
                        f"{self._process_name} method '{method_name}' still has an IB disconnect after "
                        f"{elapsed:.1f}s (retry window {self._max_retry_seconds}s). "
                        "Skipping this run; the scheduler will retry on the next cycle."
                    )
                    self.data.log.critical(msg)
                    reset_ib_data_blob_and_target_state(self.target_obj)
                    return None

                next_backoff = min(backoff, MAX_BACKOFF_SECONDS)
                if elapsed + next_backoff > self._max_retry_seconds:
                    next_backoff = max(0.0, self._max_retry_seconds - elapsed)

                self.data.log.warning(
                    f"IB disconnect in {self._process_name} method '{method_name}' (attempt {attempt}, "
                    f"elapsed {elapsed:.1f}s). Reconnecting and retrying in {next_backoff:.1f}s..."
                )

                reset_ib_data_blob_and_target_state(self.target_obj)
                self._sleep_fn(next_backoff)

                attempt += 1
                backoff = min(backoff * 2.0, MAX_BACKOFF_SECONDS)
