import pytest
from util.list_sampled_contracts import (
    format_contracts_with_priced_asterisk,
    _instruments_for_system,
)


def test_format_contracts_with_priced_asterisk():
    contracts = ["20260300", "20260600", "20260900", "20261200"]

    # When priced contract matches
    formatted = format_contracts_with_priced_asterisk(contracts, "20260600")
    assert formatted == ["20260300", "20260600*", "20260900", "20261200"]

    # When priced contract is not in list
    formatted_none_match = format_contracts_with_priced_asterisk(contracts, "20251200")
    assert formatted_none_match == ["20260300", "20260600", "20260900", "20261200"]

    # When priced_contract_id is None
    formatted_none = format_contracts_with_priced_asterisk(contracts, None)
    assert formatted_none == ["20260300", "20260600", "20260900", "20261200"]


def test_instruments_for_system():
    # Load from actual repo system config
    insts = _instruments_for_system("systems/hdenman/production/system.yaml")
    assert len(insts) == 8
    assert "SP500_micro" in insts
    assert "EUROSTX" in insts

    # Non-existent system returns empty list
    invalid = _instruments_for_system("nonexistent.system.yaml")
    assert invalid == []
