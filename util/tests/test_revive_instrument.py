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


def test_revive_instrument_seed_from_ib_executes():
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
        "util.revive_instrument.update_multiple_adjusted_prices_for_instrument"
    ):
        res = revive_instrument("TEST", options)
        assert res.seeded_prices_from_ib is True
        mock_seed.assert_called_once_with("TEST", fill_gaps_only=True)
