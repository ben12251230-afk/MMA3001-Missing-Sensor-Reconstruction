import numpy as np
import pandas as pd

from src.validation import (
    create_missing_windows,
    validate_reconstruction,
    endpoint_constrained_combined_regression
)


def make_test_data():
    """Create a small, regularly sampled temperature dataset."""

    x = np.arange(60)

    return pd.DataFrame({
        "time": pd.date_range(
            "2024-05-25", periods=60, freq="10min"
        ),
        "temperature": 20 + 0.05 * x,
        "humidity": 45 + np.sin(x / 5),
        "pressure": 1010 + np.cos(x / 6)
    })


def test_create_missing_windows():
    """Check the number and duration of test windows."""

    data = make_test_data()

    windows = create_missing_windows(
        data,
        gap_durations=(10, 30, 60, 120),
        n_tests=2
    )

    assert len(windows) == 8

    assert set(windows["duration_min"]) == {
        10, 30, 60, 120
    }


def test_validate_reconstruction():
    """Check that all six methods produce valid results."""

    data = make_test_data()

    windows = create_missing_windows(
        data,
        gap_durations=(10, 30),
        n_tests=2
    )

    # Simple mock model for testing the validation workflow
    class MockModel:
        def predict(self, X):
            return np.full(len(X), 22.0)

    model = MockModel()

    results = validate_reconstruction(
        validation_data=data,
        missing_windows=windows,
        humidity_model=model,
        pressure_model=model,
        combined_model=model,
        temperature_range=3.0
    )

    # 4 windows × 7 methods = 28 evaluations
    assert len(results) == 28

    # Check that the new method was evaluated
    assert "Endpoint-constrained regression" in results["method"].values

    # All calculated errors must be valid and non-negative
    assert results[["MAE", "RMSE", "NRMSE (%)"]].notna().all().all()
    assert (results[["MAE", "RMSE", "NRMSE (%)"]] >= 0).all().all()

    # Linear interpolation should reconstruct a linear trend
    linear_results = results[results["method"] == "Linear"]

    assert np.allclose(linear_results["RMSE"], 0, atol=1e-9)

def test_endpoint_constraints():
    """Verify that reconstructed temperatures match both endpoints."""

    data = make_test_data()

    before = data.iloc[:5]
    after = data.iloc[10:14]

    # Include both boundary rows to test the mathematical constraint
    test_points = pd.concat([
        before.tail(1),
        data.iloc[5:10],
        after.head(1)
    ]).reset_index(drop=True)

    class MockModel:
        def predict(self, X):
            return (
                18.0
                + 0.02 * X["humidity"].to_numpy()
                + 0.005 * X["pressure"].to_numpy()
            )

    predictions = endpoint_constrained_combined_regression(
        before=before,
        hidden=test_points,
        after=after,
        combined_model=MockModel()
    )

    # Check both boundary constraints
    assert np.isclose(
        predictions[0],
        before["temperature"].iloc[-1]
    )

    assert np.isclose(
        predictions[-1],
        after["temperature"].iloc[0]
    )