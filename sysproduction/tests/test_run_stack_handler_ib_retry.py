import unittest
from unittest.mock import MagicMock, patch
import time

from sysproduction.run_stack_handler import (
    RobustIBStackHandler,
    is_ib_disconnect_exception,
    reset_ib_data_blob_and_handler_state,
)


class TestRunStackHandlerIBRetry(unittest.TestCase):
    def test_is_ib_disconnect_exception(self):
        self.assertTrue(is_ib_disconnect_exception(ConnectionError("Socket disconnect")))
        self.assertTrue(is_ib_disconnect_exception(Exception("Peer closed connection.")))
        self.assertFalse(is_ib_disconnect_exception(ValueError("Invalid value")))
        self.assertFalse(is_ib_disconnect_exception(KeyError("missing_key")))

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
        self.assertTrue(mock_stack_handler.data.log.warning.called)

    def test_reset_ib_data_blob_and_handler_state(self):
        mock_stack_handler = MagicMock()
        data = MagicMock()
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

    def test_stuck_for_15min_logs_critical_and_raises(self):
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
            with self.assertRaises(ConnectionError):
                robust.check_external_position_break()

        self.assertTrue(mock_stack_handler.data.log.critical.called)
        critical_msg = mock_stack_handler.data.log.critical.call_args[0][0]
        self.assertIn("check_external_position_break", critical_msg)
        self.assertIn("exceeded limit of 900", critical_msg)


if __name__ == "__main__":
    unittest.main()
