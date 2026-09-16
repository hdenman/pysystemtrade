from syscontrol.run_process import processToRun
from sysexecution.stack_handler.stack_handler import stackHandler
from sysdata.data_blob import dataBlob
from sysbrokers.IB.ib_retry import RobustIBRetryWrapper
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
    robust_stack_handler = RobustIBRetryWrapper(
        stack_handler, process_name="run_stack_handler"
    )

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
