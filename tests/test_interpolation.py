import numpy as np
import pandas as pd

from src.interpolation import (
    create_missing_gap,
    linear_interpolation,
    cubic_spline_interpolation
)


def make_test_data():
    """Create a simple temperature dataset for testing."""
    return pd.DataFrame({
        "time": pd.date_range(
            "2024-01-01 10:00",
            periods=6,
            freq="10min"
        ),
        "temperature": [
            20.0,
            21.0,
            22.0,
            23.0,
            24.0,
            25.0
        ]
    })


def test_create_missing_gap():
    data = make_test_data()

    result = create_missing_gap(
        data,
        gap_size=2,
        start_idx=2
    )

    assert result["temperature"].isna().sum() == 2
    assert result.loc[2, "actual"] == 22.0
    assert result.loc[3, "actual"] == 23.0


def test_linear_interpolation():
    data = make_test_data()

    result = create_missing_gap(
        data,
        gap_size=2,
        start_idx=2
    )

    result = linear_interpolation(result)

    assert np.isclose(
        result.loc[2, "linear"],
        22.0
    )

    assert np.isclose(
        result.loc[3, "linear"],
        23.0
    )


def test_cubic_spline_interpolation():
    data = make_test_data()

    result = create_missing_gap(
        data,
        gap_size=2,
        start_idx=2
    )

    result = cubic_spline_interpolation(result)

    assert np.isclose(
        result.loc[2, "cubic"],
        22.0
    )

    assert np.isclose(
        result.loc[3, "cubic"],
        23.0
    )