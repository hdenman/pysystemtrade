"""
We create adjusted prices using multiple prices stored in database

We then store those adjusted prices in database and/or csv

"""
from syscore.constants import arg_not_supplied
from sysdata.csv.csv_adjusted_prices import csvFuturesAdjustedPricesData

from sysobjects.adjusted_prices import futuresAdjustedPrices

from sysproduction.data.prices import diagPrices, get_valid_instrument_code_from_user

diag_prices = diagPrices()


def _get_data_inputs(csv_adj_data_path):
    db_multiple_prices = diag_prices.db_futures_multiple_prices_data
    db_adjusted_prices = diag_prices.db_futures_adjusted_prices_data
    csv_adjusted_prices = csvFuturesAdjustedPricesData(csv_adj_data_path)

    return db_multiple_prices, db_adjusted_prices, csv_adjusted_prices


def process_adjusted_prices_all_instruments(
    csv_adj_data_path=arg_not_supplied, ADD_TO_DB=True, ADD_TO_CSV=False
):
    db_multiple_prices, _notused, _alsonotused = _get_data_inputs(csv_adj_data_path)
    instrument_list = db_multiple_prices.get_list_of_instruments()
    for instrument_code in instrument_list:
        print(instrument_code)
        process_adjusted_prices_single_instrument(
            instrument_code,
            csv_adj_data_path=csv_adj_data_path,
            ADD_TO_DB=ADD_TO_DB,
            ADD_TO_CSV=ADD_TO_CSV,
        )


def process_adjusted_prices_single_instrument(
    instrument_code,
    csv_adj_data_path=arg_not_supplied,
    multiple_prices=arg_not_supplied,
    ADD_TO_DB=True,
    ADD_TO_CSV=False,
):
    (
        db_multiple_prices,
        db_adjusted_prices,
        csv_adjusted_prices,
    ) = _get_data_inputs(csv_adj_data_path)
    if multiple_prices is arg_not_supplied:
        multiple_prices = db_multiple_prices.get_multiple_prices(instrument_code)
    adjusted_prices = futuresAdjustedPrices.stitch_multiple_prices(
        multiple_prices, forward_fill=True
    )

    print(adjusted_prices)

    if ADD_TO_DB:
        db_adjusted_prices.add_adjusted_prices(
            instrument_code, adjusted_prices, ignore_duplication=True
        )
    if ADD_TO_CSV:
        csv_adjusted_prices.add_adjusted_prices(
            instrument_code, adjusted_prices, ignore_duplication=True
        )

    return adjusted_prices


def bulk_update():
    input("Will overwrite existing prices. CTL-C to abort.")
    process_adjusted_prices_all_instruments(
        csv_adj_data_path=arg_not_supplied, ADD_TO_DB=True, ADD_TO_CSV=False
    )


def individual_update():
    print("Will overwrite existing prices. CTL-C to abort.")
    do_another = True
    while do_another:
        EXIT_STR = "Finished: Exit"
        instrument_code = get_valid_instrument_code_from_user(
            source="single", allow_exit=True, exit_code=EXIT_STR
        )
        if instrument_code is EXIT_STR:
            do_another = False
        else:
            process_adjusted_prices_single_instrument(
                instrument_code,
                csv_adj_data_path=arg_not_supplied,
                ADD_TO_DB=True,
                ADD_TO_CSV=False,
            )


if __name__ == "__main__":
    # bulk_update()
    individual_update()
