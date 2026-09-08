import pandas as pd
from util.check_instrument import _gap_check, FAIL

def test_gap_check_most_recent_first():
    # Dates: gap 1 in 2020, gap 2 in 2025
    dr1 = pd.date_range("2020-01-01", "2020-01-05")
    dr2 = pd.date_range("2020-01-10", "2025-01-01")
    dr3 = pd.date_range("2025-01-10", "2025-01-15")
    dates = pd.DatetimeIndex(dr1).union(pd.DatetimeIndex(dr2)).union(pd.DatetimeIndex(dr3))
    series = pd.Series(1.0, index=dates)
    today = pd.Timestamp(2025, 1, 15)

    result = _gap_check(series, "Test Series", today)
    assert result.status == FAIL
    assert len(result.notes) >= 3
    # Check that note lists the most recent gap (2025) before the older gap (2020)
    gap_notes = [n for n in result.notes if "gap:" in n]
    assert "2025" in gap_notes[0]
    assert "2020" in gap_notes[1]
