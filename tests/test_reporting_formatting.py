import pandas as pd
from sysproduction.reporting.formatting import nice_format_slippage_table


def test_nice_format_slippage_table_with_object_dtype_floats():
    df = pd.DataFrame(
        {
            "Difference": [12.34, -98.76],
            "bid_ask_trades": [0.12346, 0.54321],
            "total_trades": [0.12346, 0.54321],
            "bid_ask_sampled": [0.12346, 0.54321],
            "weight_trades": [0.3333, 0.6666],
            "weight_samples": [0.3333, 0.6666],
            "weight_config": [0.3334, 0.6667],
            "estimate": [0.12346, 0.54321],
            "Configured": [0.12346, 0.54321],
        },
        index=["EDOLLAR", "US10"],
    ).astype(object)

    result = nice_format_slippage_table(df.copy())

    # Assert observable rounded values
    assert list(result["Difference"]) == [12.3, -98.8]
    assert list(result["bid_ask_trades"]) == [0.1235, 0.5432]
    assert list(result["total_trades"]) == [0.1235, 0.5432]
    assert list(result["bid_ask_sampled"]) == [0.1235, 0.5432]
    assert list(result["weight_trades"]) == [0.33, 0.67]
    assert list(result["weight_samples"]) == [0.33, 0.67]
    assert list(result["weight_config"]) == [0.33, 0.67]
    assert list(result["estimate"]) == [0.1235, 0.5432]
    assert list(result["Configured"]) == [0.1235, 0.5432]

    # Assert columns were coerced to numeric dtype
    for col in df.columns:
        assert pd.api.types.is_numeric_dtype(result[col])
