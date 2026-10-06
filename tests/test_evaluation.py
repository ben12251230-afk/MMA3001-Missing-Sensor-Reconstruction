import numpy as np

from src.evaluation import calculate_errors


def test_calculate_errors_perfect_prediction():
    actual = [20.0, 21.0, 22.0]
    predicted = [20.0, 21.0, 22.0]

    mae, rmse = calculate_errors(
        actual,
        predicted
    )

    assert np.isclose(mae, 0.0)
    assert np.isclose(rmse, 0.0)


def test_calculate_errors_known_values():
    actual = [20.0, 22.0]
    predicted = [21.0, 21.0]

    mae, rmse = calculate_errors(
        actual,
        predicted
    )

    assert np.isclose(mae, 1.0)
    assert np.isclose(rmse, 1.0)