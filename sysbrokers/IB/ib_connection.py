"""
IB connection using ib_async https://github.com/ib-api-reloaded/ib_async

"""

import time

from ib_async import IB

from sysbrokers.IB.ib_connection_defaults import ib_defaults
from syscore.exceptions import missingData
from syscore.constants import arg_not_supplied

from syslogging.logger import *

from sysdata.config.production_config import get_production_config


IB_ERROR_CONNECTIVITY_LOST = 1100
IB_ERROR_CONNECTIVITY_RESTORED = {1101, 1102}
IB_REQUEST_TIMEOUT_SECONDS = 15 * 60



class connectionIB(object):
    """
    Connection object for connecting IB
    (A database plug in will need to be added for streaming prices)
    """

    def __init__(
        self,
        client_id: int,
        ib_ipaddress: str = arg_not_supplied,
        ib_port: int = arg_not_supplied,
        account: str = arg_not_supplied,
        log_name: str = "connectionIB",
    ):
        """
        :param client_id: client id
        :param ipaddress: IP address of machine running IB Gateway or TWS. If not passed then will get from private config file, or defaults
        :param port: Port listened to by IB Gateway or TWS
        :param log_name: calling log name
        :param mongo_db: mongoDB connection
        """

        # resolve defaults
        ipaddress, port, __ = ib_defaults(ib_ipaddress=ib_ipaddress, ib_port=ib_port)
        self._ib_connection_config = dict(
            ipaddress=ipaddress, port=port, client=client_id
        )

        # The client id is pulled from a mongo database
        # If for example you want to use a different database you could do something like:
        # connectionIB(mongo_ib_tracker =
        # mongoIBclientIDtracker(database_name="another")

        # If you copy for another broker include these lines
        self._log = get_logger(
            "connectionIB",
            {
                TYPE_LOG_LABEL: log_name,
                BROKER_LOG_LABEL: "IB",
                CLIENTID_LOG_LABEL: client_id,
            },
        )
        self._ib_server_connected = True


        # You can pass a client id yourself, or let IB find one

        try:
            self._init_connection(
                ipaddress=ipaddress, port=port, client_id=client_id, account=account
            )
        except Exception as e:
            # Log all exceptions generated during connection as critical error.
            # Under the default production setup this should send an email.
            # Error is reraised as we can't really continue and user intervention is required
            self.log.critical(
                f"IB connection failed with exception - {e}, connection aborted."
            )
            raise

    def _init_connection(
        self, ipaddress: str, port: int, client_id: int, account=arg_not_supplied
    ):
        ib = IB()
        ib.RequestTimeout = IB_REQUEST_TIMEOUT_SECONDS
        ib.errorEvent += self._handle_ib_error_event

        try:
            if account is arg_not_supplied:
                ## not passed get from config
                account = get_broker_account()
        except missingData:
            self.log.error(
                "Broker account ID not found in private config - may cause issues"
            )
            ib.connect(ipaddress, port, clientId=client_id)
        else:
            ## connect using account
            ib.connect(ipaddress, port, clientId=client_id, account=account)

        # Sometimes takes a few seconds to resolve... only have to do this once per process so no biggie
        time.sleep(5)

        self._ib = ib
        self._account = account
        self._ib_server_connected = True

    def _handle_ib_error_event(self, req_id, error_code, error_string, contract):
        if error_code == IB_ERROR_CONNECTIVITY_LOST:
            self._ib_server_connected = False
        elif error_code in IB_ERROR_CONNECTIVITY_RESTORED:
            self._ib_server_connected = True

    def has_active_connection_problem(self) -> bool:
        return not self.ib.isConnected() or not self._ib_server_connected

    def connection_problem_description(self) -> str:
        if not self.ib.isConnected():
            return "IB API socket is disconnected"
        if not self._ib_server_connected:
            return "IBKR connectivity from Trader Workstation/Gateway is lost"
        return "IB connection is healthy"

    def remove_event_handlers(self):
        try:
            self.ib.errorEvent -= self._handle_ib_error_event
        except Exception:
            pass

    @property
    def ib(self):
        return self._ib

    @property
    def log(self):
        return self._log

    def __repr__(self):
        return "IB broker connection" + str(self._ib_connection_config)

    def client_id(self):
        return self._ib_connection_config["client"]

    @property
    def account(self):
        return self._account

    def close_connection(self):
        self.log.debug("Terminating %s" % str(self._ib_connection_config))
        try:
            self.remove_event_handlers()
            # Try and disconnect IB client
            self.ib.disconnect()
        except BaseException:
            self.log.warning(
                "Trying to disconnect IB client failed... ensure process is killed"
            )

def get_broker_account() -> str:
    production_config = get_production_config()
    account_id = production_config.get_element("broker_account")
    return account_id
