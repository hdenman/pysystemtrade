import errno
import unittest
from unittest.mock import MagicMock, patch
import time
from types import SimpleNamespace

from sysexecution.algos.common_functions import raise_if_active_broker_connection_problem
from sysbrokers.IB.ib_connection import (
    IB_ERROR_CONNECTIVITY_LOST,
    IB_ERROR_CONNECTIVITY_RESTORED,
    IB_REQUEST_TIMEOUT_SECONDS,
    connectionIB,
)
from sysproduction.run_stack_handler import (
    RobustIBStackHandler,
    is_ib_disconnect_exception,
    raise_if_known_ib_connection_problem,
    reset_ib_data_blob_and_handler_state,
)
from syscore.constants import arg_not_supplied
from sysdata.data_blob import dataBlob

def _make_wrapped_timeout_error():
    try:
        raise TimeoutError("Connection attempt timed out")
    except TimeoutError:
        raise Exception("Error  couldn't evaluate ibFxPricesData(self.ib_conn, self) This might be because (a) IB gateway not running")

class TestRunStackHandlerIBRetry(unittest.TestCase):
    def test_is_ib_disconnect_exception(self):
        self.assertTrue(is_ib_disconnect_exception(ConnectionError("Socket disconnect")))
        self.assertTrue(is_ib_disconnect_exception(Exception("Peer closed connection.")))
        self.assertTrue(is_ib_disconnect_exception(TimeoutError()))
        self.assertTrue(is_ib_disconnect_exception(Exception("API connection failed: TimeoutError()")))
        self.assertTrue(is_ib_disconnect_exception(OSError(errno.ETIMEDOUT, "connection timed out")))
        try:
            _make_wrapped_timeout_error()
        except Exception as exc:
            self.assertTrue(is_ib_disconnect_exception(exc))
        self.assertFalse(
            is_ib_disconnect_exception(
                Exception(
                    "Error  couldn't evaluate ibFxPricesData(self.ib_conn, self) "
                    "This might be because import is missing"
                )
            )
        )
        self.assertFalse(is_ib_disconnect_exception(ValueError("Invalid value")))
        self.assertFalse(is_ib_disconnect_exception(KeyError("missing_key")))

    def test_ib_error_1100_marks_connection_unhealthy_until_restored(self):
        conn = object.__new__(connectionIB)
        conn._ib = SimpleNamespace(isConnected=MagicMock(return_value=True))
        conn._ib_server_connected = True

        conn._handle_ib_error_event(
            -1,
            IB_ERROR_CONNECTIVITY_LOST,
            "Connectivity between IBKR and Trader Workstation has been lost.",
            None,
        )
        self.assertTrue(conn.has_active_connection_problem())
        self.assertIn("IBKR connectivity", conn.connection_problem_description())

        conn._handle_ib_error_event(
            -1,
            next(iter(IB_ERROR_CONNECTIVITY_RESTORED)),
            "Connectivity between IBKR and Trader Workstation has been restored.",
            None,
        )
        self.assertFalse(conn.has_active_connection_problem())

    def test_ib_connection_sets_blocking_request_timeout(self):
        conn = object.__new__(connectionIB)
        ib = MagicMock()

        with patch("sysbrokers.IB.ib_connection.IB", return_value=ib):
            with patch("sysbrokers.IB.ib_connection.time.sleep"):
                conn._init_connection("127.0.0.1", 4001, 1, account="DU123")

        self.assertEqual(ib.RequestTimeout, IB_REQUEST_TIMEOUT_SECONDS)
        ib.connect.assert_called_once_with(
            "127.0.0.1", 4001, clientId=1, account="DU123"
        )

    def test_existing_unhealthy_ib_connection_is_reconnected_before_method(self):
        mock_stack_handler = MagicMock()
        ib_conn = MagicMock()
        ib_conn.has_active_connection_problem.return_value = True
        ib_conn.connection_problem_description.return_value = "IBKR connectivity from Trader Workstation/Gateway is lost"
        mock_stack_handler.data._ib_conn = ib_conn
        mock_stack_handler.data.log = MagicMock()
        mock_stack_handler.check_external_position_break.return_value = "success"

        times = [0.0, 1.0, 2.0]
        def fake_time():
            return times.pop(0) if times else 3.0

        robust = RobustIBStackHandler(
            mock_stack_handler,
            max_retry_seconds=900,
            time_fn=fake_time,
            sleep_fn=MagicMock(),
        )

        with patch("sysproduction.run_stack_handler.reset_ib_data_blob_and_handler_state") as mock_reset:
            mock_reset.side_effect = lambda stack_handler: setattr(
                stack_handler.data, "_ib_conn", arg_not_supplied
            )
            result = robust.check_external_position_break()

        self.assertEqual(result, "success")
        mock_reset.assert_called_once_with(mock_stack_handler)

    def test_connection_problem_checks_ignore_no_connection(self):
        data = SimpleNamespace(_ib_conn=arg_not_supplied)
        self.assertIsNone(raise_if_known_ib_connection_problem(data))
        self.assertIsNone(raise_if_active_broker_connection_problem(data))

    def test_algo_loop_health_check_raises_for_unhealthy_connection(self):
        ib_conn = MagicMock()
        ib_conn.has_active_connection_problem.return_value = True
        ib_conn.connection_problem_description.return_value = "IB API socket is disconnected"
        data = SimpleNamespace(_ib_conn=ib_conn)

        with self.assertRaises(ConnectionError):
            raise_if_active_broker_connection_problem(data)

    def test_successful_execution_no_retry(self):
        mock_stack_handler = MagicMock()
        mock_stack_handler.data.log = MagicMock()
        mock_stack_handler.some_method.return_value = "ok"

        robust = RobustIBStackHandler(
            mock_stack_handler,
            max_retry_seconds=900,
            time_fn=time.monotonic,
            sleep_fn=MagicMock(),
        )

        result = robust.some_method()
        self.assertEqual(result, "ok")
        self.assertEqual(mock_stack_handler.some_method.call_count, 1)

    def test_transient_ib_disconnect_retries_and_succeeds(self):
        mock_stack_handler = MagicMock()
        mock_stack_handler.data.log = MagicMock()

        # Fail twice with ConnectionError("Socket disconnect"), then succeed
        mock_stack_handler.check_external_position_break.side_effect = [
            ConnectionError("Socket disconnect"),
            ConnectionError("Socket disconnect"),
            "success",
        ]

        times = [0.0, 1.0, 3.0, 5.0]
        def fake_time():
            return times.pop(0) if times else 10.0

        sleep_calls = []
        def fake_sleep(secs):
            sleep_calls.append(secs)

        robust = RobustIBStackHandler(
            mock_stack_handler,
            max_retry_seconds=900,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        with patch("sysproduction.run_stack_handler.reset_ib_data_blob_and_handler_state") as mock_reset:
            result = robust.check_external_position_break()

        self.assertEqual(result, "success")
        self.assertEqual(mock_stack_handler.check_external_position_break.call_count, 3)
        self.assertEqual(mock_reset.call_count, 2)
        self.assertEqual(sleep_calls, [1.0, 2.0])
    def test_wrapped_timeout_retries_and_succeeds(self):
        mock_stack_handler = MagicMock()
        mock_stack_handler.data.log = MagicMock()

        wrapped_exc = None
        try:
            _make_wrapped_timeout_error()
        except Exception as exc:
            wrapped_exc = exc

        mock_stack_handler.check_external_position_break.side_effect = [
            wrapped_exc,
            "success",
        ]

        times = [0.0, 1.0, 2.0]
        def fake_time():
            return times.pop(0) if times else 10.0

        sleep_calls = []
        def fake_sleep(secs):
            sleep_calls.append(secs)

        robust = RobustIBStackHandler(
            mock_stack_handler,
            max_retry_seconds=900,
            time_fn=fake_time,
            sleep_fn=fake_sleep,
        )

        with patch("sysproduction.run_stack_handler.reset_ib_data_blob_and_handler_state") as mock_reset:
            result = robust.check_external_position_break()

        self.assertEqual(result, "success")
        self.assertEqual(mock_stack_handler.check_external_position_break.call_count, 2)
        self.assertEqual(mock_reset.call_count, 1)
        self.assertEqual(sleep_calls, [1.0])
        self.assertTrue(mock_stack_handler.data.log.warning.called)
        self.assertTrue(mock_stack_handler.data.log.warning.called)

    def test_reset_ib_data_blob_and_handler_state(self):
        mock_stack_handler = MagicMock()
        data = SimpleNamespace(
            log=MagicMock(),
            db_ib_broker_client_id=MagicMock(),
            _get_new_name=MagicMock(return_value="broker_contract_position"),
        )
        mock_stack_handler.data = data
        mock_stack_handler._data_broker = MagicMock()

        ib_conn = MagicMock()
        ib_conn.client_id.return_value = 123
        data._ib_conn = ib_conn
        data._attr_list = ["broker_contract_position"]

        with patch("sysproduction.run_stack_handler.get_broker_class_list", return_value=[MagicMock(__name__="brokerContractPositionData")]):
            with patch("sysproduction.run_stack_handler.get_class_name", return_value="brokerContractPositionData"):
                data._get_new_name.return_value = "broker_contract_position"
                setattr(data, "broker_contract_position", MagicMock())

                reset_ib_data_blob_and_handler_state(mock_stack_handler)

        ib_conn.close_connection.assert_called_once()
        data.db_ib_broker_client_id.release_clientid.assert_called_once_with(123)
        self.assertFalse(hasattr(data, "broker_contract_position"))
        self.assertFalse(hasattr(mock_stack_handler, "_data_broker"))
        self.assertIs(data._ib_conn, arg_not_supplied)

    def test_reset_converts_none_ib_connection_to_arg_not_supplied(self):
        mock_stack_handler = MagicMock()
        data = SimpleNamespace(
            log=MagicMock(),
            db_ib_broker_client_id=MagicMock(),
            _get_new_name=MagicMock(),
            _ib_conn=None,
            _attr_list=[],
        )
        mock_stack_handler.data = data

        with patch("sysproduction.run_stack_handler.get_broker_class_list", return_value=[]):
            reset_ib_data_blob_and_handler_state(mock_stack_handler)

        self.assertIs(data._ib_conn, arg_not_supplied)
        data.db_ib_broker_client_id.release_clientid.assert_not_called()

    def test_reset_allows_data_blob_to_create_new_ib_connection(self):
        mock_stack_handler = MagicMock()
        old_conn = MagicMock()
        old_conn.client_id.return_value = 123
        new_conn = MagicMock()
        log = MagicMock()
        log.name = "stack_handler"
        data = dataBlob(log=log, ib_conn=old_conn)
        data.db_ib_broker_client_id = MagicMock()
        mock_stack_handler.data = data

        with patch("sysproduction.run_stack_handler.get_broker_class_list", return_value=[]):
            reset_ib_data_blob_and_handler_state(mock_stack_handler)

        with patch.object(data, "_get_new_ib_connection", return_value=new_conn) as get_new_ib_connection:
            self.assertIs(data.ib_conn, new_conn)

        get_new_ib_connection.assert_called_once_with()

    def test_non_ib_exception_does_not_retry(self):
        mock_stack_handler = MagicMock()
        mock_stack_handler.data.log = MagicMock()
        mock_stack_handler.check_external_position_break.side_effect = ValueError("Logic error")

        robust = RobustIBStackHandler(
            mock_stack_handler,
            max_retry_seconds=900,
            time_fn=time.monotonic,
            sleep_fn=MagicMock(),
        )

        with patch("sysproduction.run_stack_handler.reset_ib_data_blob_and_handler_state") as mock_reset:
            with self.assertRaises(ValueError):
                robust.check_external_position_break()

        self.assertEqual(mock_stack_handler.check_external_position_break.call_count, 1)
        mock_reset.assert_not_called()

    def test_stuck_for_15min_logs_critical_and_skips_run(self):
        mock_stack_handler = MagicMock()
        mock_stack_handler.data.log = MagicMock()
        mock_stack_handler.check_external_position_break.side_effect = ConnectionError("Socket disconnect")

        times = [0.0, 100.0, 950.0]
        def fake_time():
            return times.pop(0) if times else 1000.0

        robust = RobustIBStackHandler(
            mock_stack_handler,
            max_retry_seconds=900,
            time_fn=fake_time,
            sleep_fn=MagicMock(),
        )

        with patch("sysproduction.run_stack_handler.reset_ib_data_blob_and_handler_state"):
            result = robust.check_external_position_break()

        self.assertIsNone(result)
        self.assertTrue(mock_stack_handler.data.log.critical.called)
        critical_msg = mock_stack_handler.data.log.critical.call_args[0][0]
        self.assertIn("check_external_position_break", critical_msg)
        self.assertIn("retry window 900", critical_msg)
        self.assertIn("Skipping this run", critical_msg)


if __name__ == "__main__":
    unittest.main()
