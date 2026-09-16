from sysproduction.data.prices import diagPrices
from syscore.exceptions import missingData
from sysbrokers.IB.ib_futures_contract_price_data import (
    futuresContract,
)
from syscore.dateutils import DAILY_PRICE_FREQ, HOURLY_FREQ, Frequency
from sysdata.data_blob import dataBlob

from sysproduction.data.broker import dataBroker
from sysproduction.data.prices import updatePrices
from sysproduction.update_historical_prices import write_merged_prices_for_contract


def seed_price_data_from_IB(instrument_code: str, fill_gaps_only: bool = True):
    data = dataBlob()
    data_broker = dataBroker(data)
    diag_prices = diagPrices(data)

    existing_contract_dates = set(
        diag_prices.contract_dates_with_price_data_for_instrument_code(
            instrument_code
        )
    )

    list_of_contracts = data_broker.get_list_of_contract_dates_for_instrument_code(
        instrument_code, allow_expired=True
    )

    data.log.info(
        "Seeding price data from IB for %s (found %d contract dates from broker, %d existing in DB)"
        % (instrument_code, len(list_of_contracts), len(existing_contract_dates))
    )

    ## This returns yyyymmdd strings, where we have the actual expiry date

    for contract_date in list_of_contracts:
        ## We do this slightly tortuous thing because there are energy contracts
        ## which don't expire in the month they are labelled with
        ## So for example, CRUDE_W 202106 actually expires on 20210528

        date_str = contract_date[:6]
        contract_object = futuresContract(instrument_code, date_str)

        if fill_gaps_only and date_str in existing_contract_dates:
            data.log.info(
                "Skipping seeding for %s (already has price data in DB)"
                % str(contract_object)
            )
            continue

        seed_price_data_for_contract(data=data, contract_object=contract_object)


def seed_price_data_for_contract(data: dataBlob, contract_object: futuresContract):
    log_attrs = {**contract_object.log_attributes(), "method": "temp"}

    list_of_frequencies = [HOURLY_FREQ, DAILY_PRICE_FREQ]
    for frequency in list_of_frequencies:
        data.log.debug("Getting data at frequency %s" % str(frequency), **log_attrs)
        seed_price_data_for_contract_at_frequency(
            data=data, contract_object=contract_object, frequency=frequency
        )

    data.log.debug("Writing merged data for %s" % str(contract_object), **log_attrs)
    write_merged_prices_for_contract(
        data, contract_object=contract_object, list_of_frequencies=list_of_frequencies
    )


def seed_price_data_for_contract_at_frequency(
    data: dataBlob, contract_object: futuresContract, frequency: Frequency
):
    data_broker = dataBroker(data)
    update_prices = updatePrices(data)
    log_attrs = {**contract_object.log_attributes(), "method": "temp"}

    try:
        prices = (
            data_broker.get_prices_at_frequency_for_potentially_expired_contract_object(
                contract_object, frequency=frequency
            )
        )
    except missingData:
        data.log.warning(
            "Error getting data for %s" % str(contract_object),
            **log_attrs,
        )
        return None

    data.log.debug(
        "Got %d lines of prices for %s" % (len(prices), str(contract_object)),
        **log_attrs,
    )

    if len(prices) == 0:
        data.log.warning(
            "No price data for %s" % str(contract_object),
            **log_attrs,
        )
    else:
        update_prices.overwrite_prices_at_frequency_for_contract(
            contract_object=contract_object, frequency=frequency, new_prices=prices
        )


if __name__ == "__main__":
    print("Get initial price data from IB")
    instrument_code = input("Instrument code? <return to abort> ")
    if instrument_code == "":
        exit()

    seed_price_data_from_IB(instrument_code)
