import pandas as pd

from syscore.pandas.merge_data_with_label_column import merge_data_series_with_label_column


def test_merge_data_series_uses_original_label_before_gap():
    original = pd.DataFrame(
        {
            "PRICE": [1.0, 2.0],
            "PRICE_CONTRACT": ["old", "current"],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    new = pd.DataFrame(
        {
            "PRICE": [3.0, 4.0],
            "PRICE_CONTRACT": ["current", "current"],
        },
        index=pd.to_datetime(["2024-06-01", "2024-06-02"]),
    )

    merged = merge_data_series_with_label_column(
        original,
        new,
        data_column="PRICE",
        label_column="PRICE_CONTRACT",
    )

    assert list(merged.index) == list(original.index) + list(new.index)
    assert merged.loc[pd.Timestamp("2024-06-01"), "PRICE"] == 3.0
    assert merged.loc[pd.Timestamp("2024-06-02"), "PRICE"] == 4.0
    assert (
        merged.loc[pd.Timestamp("2024-06-02"), "PRICE_CONTRACT"]
        == "current"
    )
