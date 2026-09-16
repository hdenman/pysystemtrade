import datetime
from unittest.mock import MagicMock, patch

from util.revive_instrument import ReviveOptions, revive_instrument


def test_revive_instrument_seed_from_ib_dry_run():
    options = ReviveOptions(
        as_of_date=datetime.date(2025, 1, 1),
        seed_from_ib=True,
        dry_run=True,
    )
    with patch("util.revive_instrument.seed_price_data_from_IB") as mock_seed, patch(
        "util.revive_instrument._current_priced_contract", return_value="20241200"
    ), patch(
        "util.revive_instrument._roll_until_priced_contract_is_live", return_value=0
    ):
        res = revive_instrument("TEST", options)
        assert res.seeded_prices_from_ib is True
        mock_seed.assert_not_called()


def test_revive_instrument_seed_from_ib_executes_and_rebuilds_history():
    options = ReviveOptions(
        as_of_date=datetime.date(2025, 1, 1),
        seed_from_ib=True,
        dry_run=False,
        skip_sampled_contracts=True,
        skip_price_download=True,
    )
    with patch("util.revive_instrument.seed_price_data_from_IB") as mock_seed, patch(
        "util.revive_instrument._current_priced_contract", return_value="20241200"
    ), patch(
        "util.revive_instrument._roll_until_priced_contract_is_live", return_value=0
    ), patch(
        "util.revive_instrument.build_and_write_roll_calendar"
    ) as mock_build_calendar, patch(
        "util.revive_instrument.process_multiple_prices_single_instrument"
    ) as mock_multiple, patch(
        "util.revive_instrument.process_adjusted_prices_single_instrument"
    ) as mock_adjusted:
        res = revive_instrument("TEST", options)
        assert res.seeded_prices_from_ib is True
        assert res.rebuilt_history is True
        mock_seed.assert_called_once_with("TEST", fill_gaps_only=True)
        mock_build_calendar.assert_called_once_with("TEST", write=True, check_before_writing=False)
        mock_multiple.assert_called_once_with("TEST", ADD_TO_DB=True, ADD_TO_CSV=False)
        mock_adjusted.assert_called_once_with("TEST", multiple_prices=mock_multiple.return_value, ADD_TO_DB=True, ADD_TO_CSV=False)


def test_revive_instrument_no_rebuild_history_override():
    options = ReviveOptions(
        as_of_date=datetime.date(2025, 1, 1),
        seed_from_ib=True,
        rebuild_history=False,
        dry_run=False,
        skip_sampled_contracts=True,
        skip_price_download=True,
    )
    with patch("util.revive_instrument.seed_price_data_from_IB") as mock_seed, patch(
        "util.revive_instrument._current_priced_contract", return_value="20241200"
    ), patch(
        "util.revive_instrument._roll_until_priced_contract_is_live", return_value=0
    ), patch(
        "util.revive_instrument.update_multiple_adjusted_prices_for_instrument"
    ) as mock_update_incremental, patch(
        "util.revive_instrument.build_and_write_roll_calendar"
    ) as mock_build_calendar:
        res = revive_instrument("TEST", options)
        assert res.seeded_prices_from_ib is True
        assert res.rebuilt_history is False
        assert res.refreshed_multiple_adjusted is True
        mock_seed.assert_called_once_with("TEST", fill_gaps_only=True)
        mock_build_calendar.assert_not_called()
        mock_update_incremental.assert_called_once()
