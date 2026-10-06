import numpy as np
import pandas as pd

from .interpolation import (
    create_missing_gap,
    linear_interpolation,
    cubic_spline_interpolation
)
from .ml_model import (
    train_linear_regression,
    predict_with_regression
)


def calculate_errors(actual, predicted):
    """
    Calculate MAE and RMSE between actual and predicted values.

    Parameters
    ----------
    actual : array-like
        Ground-truth values.

    predicted : array-like
        Reconstructed values.

    Returns
    -------
    tuple
        MAE and RMSE.
    """

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    error = actual - predicted

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    return mae, rmse


def evaluate_missing_gap(data, gap_size, start_idx):
    """
    Evaluate linear and cubic spline reconstruction for one missing gap.

    Parameters
    ----------
    data : pandas.DataFrame
        Complete temperature time series.

    gap_size : int
        Number of consecutive measurements to remove.

    start_idx : int
        Starting index of the artificial missing gap.

    Returns
    -------
    dict
        Gap information and reconstruction error metrics.
    """

    experiment = create_missing_gap(
        data,
        gap_size,
        start_idx
    )

    experiment = linear_interpolation(experiment)
    experiment = cubic_spline_interpolation(experiment)

    missing_idx = list(
        range(start_idx, start_idx + gap_size)
    )

    comparison = experiment.loc[
        missing_idx,
        ["actual", "linear", "cubic"]
    ]

    linear_mae, linear_rmse = calculate_errors(
        comparison["actual"],
        comparison["linear"]
    )

    cubic_mae, cubic_rmse = calculate_errors(
        comparison["actual"],
        comparison["cubic"]
    )

    return {
        "gap_size": gap_size,
        "start_idx": start_idx,
        "linear_mae": linear_mae,
        "linear_rmse": linear_rmse,
        "cubic_mae": cubic_mae,
        "cubic_rmse": cubic_rmse
    }


def validate_multiple_blocks(
    data,
    block_ids,
    gap_sizes,
    positions_per_gap=5,
    edge_buffer=5
):
    """
    Evaluate interpolation methods across multiple continuous blocks.

    Parameters
    ----------
    data : pandas.DataFrame
        Temperature dataset containing a 'block' column.

    block_ids : list
        Continuous block IDs to test.

    gap_sizes : list
        Numbers of consecutive measurements to remove.

    positions_per_gap : int, default=5
        Number of missing-gap locations tested per block and gap size.

    edge_buffer : int, default=5
        Number of measurements kept at each end of a block.

    Returns
    -------
    pandas.DataFrame
        Results from all reconstruction experiments.
    """

    all_results = []

    for block_id in block_ids:

        block_data = (
            data[data["block"] == block_id]
            [["time", "temperature"]]
            .reset_index(drop=True)
            .copy()
        )

        for gap_size in gap_sizes:

            min_start = edge_buffer
            max_start = (
                len(block_data)
                - gap_size
                - edge_buffer
            )

            if max_start < min_start:
                continue

            start_positions = np.unique(
                np.linspace(
                    min_start,
                    max_start,
                    positions_per_gap,
                    dtype=int
                )
            )

            for start_idx in start_positions:

                result = evaluate_missing_gap(
                    block_data,
                    gap_size,
                    start_idx
                )

                result["block"] = block_id

                all_results.append(result)

    return pd.DataFrame(all_results)


def summarise_validation(results):
    """
    Summarise validation results by missing gap size.

    Parameters
    ----------
    results : pandas.DataFrame
        Output from validate_multiple_blocks().

    Returns
    -------
    pandas.DataFrame
        Mean and standard deviation of reconstruction errors.
    """

    summary = results.groupby("gap_size").agg(
        n_tests=("block", "size"),

        linear_mae_mean=("linear_mae", "mean"),
        linear_mae_std=("linear_mae", "std"),

        linear_rmse_mean=("linear_rmse", "mean"),
        linear_rmse_std=("linear_rmse", "std"),

        cubic_mae_mean=("cubic_mae", "mean"),
        cubic_mae_std=("cubic_mae", "std"),

        cubic_rmse_mean=("cubic_rmse", "mean"),
        cubic_rmse_std=("cubic_rmse", "std")
    )

    return summary

def evaluate_three_methods(
    data,
    block_id,
    gap_size,
    start_idx
):
    """
    Compare linear interpolation, cubic spline interpolation,
    and multiple linear regression for missing temperature data.
    """

    block_data = (
        data[data["block"] == block_id]
        .reset_index(drop=True)
        .copy()
    )

    # Train regression using data outside the validation block
    train_data = (
        data[data["block"] != block_id]
        .dropna(
            subset=[
                "temperature",
                "humidity",
                "pressure"
            ]
        )
        .copy()
    )

    regression_model = train_linear_regression(
        train_data,
        predictors=["humidity", "pressure"],
        target="temperature"
    )

    # Create artificial missing temperature values
    gap_data = create_missing_gap(
        block_data[["time", "temperature"]],
        gap_size=gap_size,
        start_idx=start_idx
    )

    linear_result = linear_interpolation(
        gap_data.copy()
    )

    cubic_result = cubic_spline_interpolation(
        gap_data.copy()
    )

    missing_mask = gap_data["temperature"].isna()

    actual = gap_data.loc[
        missing_mask,
        "actual"
    ].to_numpy()

    linear_pred = linear_result.loc[
        missing_mask,
        "linear"
    ].to_numpy()

    cubic_pred = cubic_result.loc[
        missing_mask,
        "cubic"
    ].to_numpy()

    regression_input = block_data.loc[
        missing_mask,
        ["humidity", "pressure"]
    ]

    regression_pred = predict_with_regression(
        regression_model,
        regression_input,
        predictors=["humidity", "pressure"]
    )

    linear_mae, linear_rmse = calculate_errors(
        actual,
        linear_pred
    )

    cubic_mae, cubic_rmse = calculate_errors(
        actual,
        cubic_pred
    )

    regression_mae, regression_rmse = calculate_errors(
        actual,
        regression_pred
    )

    return {
        "block": block_id,
        "gap_size": gap_size,
        "linear_mae": linear_mae,
        "linear_rmse": linear_rmse,
        "cubic_mae": cubic_mae,
        "cubic_rmse": cubic_rmse,
        "regression_mae": regression_mae,
        "regression_rmse": regression_rmse
    }

def validate_three_methods(
    data,
    block_ids,
    gap_sizes,
    positions_per_gap=5,
    edge_buffer=5
):
    """
    Validate three temperature reconstruction methods across
    multiple blocks, gap sizes and gap locations.
    """

    results = []

    for block_id in block_ids:

        block_data = (
            data[data["block"] == block_id]
            .reset_index(drop=True)
        )

        for gap_size in gap_sizes:

            min_start = edge_buffer

            max_start = (
                len(block_data)
                - gap_size
                - edge_buffer
            )

            if max_start < min_start:
                continue

            positions = np.unique(
                np.linspace(
                    min_start,
                    max_start,
                    positions_per_gap,
                    dtype=int
                )
            )

            for start_idx in positions:

                result = evaluate_three_methods(
                    data,
                    block_id=block_id,
                    gap_size=gap_size,
                    start_idx=start_idx
                )

                result["start_idx"] = start_idx

                results.append(result)

    return pd.DataFrame(results)


def evaluate_cross_variable_reconstruction(
    data,
    block_id,
    target,
    predictor,
    gap_size,
    start_idx
):
    """
    Compare temporal interpolation with cross-variable regression
    when humidity or pressure measurements are missing.
    """

    block_data = (
        data[data["block"] == block_id]
        .reset_index(drop=True)
        .copy()
    )

    train_data = (
        data[data["block"] != block_id]
        .dropna(
            subset=[
                target,
                predictor
            ]
        )
        .copy()
    )

    regression_model = train_linear_regression(
        train_data,
        predictors=[predictor],
        target=target
    )

    end_idx = start_idx + gap_size

    actual = block_data.loc[
        start_idx:end_idx - 1,
        target
    ].to_numpy()

    test_data = block_data[
        ["time", target, predictor]
    ].copy()

    test_data.loc[
        start_idx:end_idx - 1,
        target
    ] = np.nan

    missing_mask = test_data[target].isna()

    # Temporal interpolation
    interpolation_data = (
        test_data
        .set_index("time")
        .copy()
    )

    interpolation_data["interpolated"] = (
        interpolation_data[target]
        .interpolate(method="time")
    )

    interpolation_pred = (
        interpolation_data.loc[
            missing_mask.to_numpy(),
            "interpolated"
        ]
        .to_numpy()
    )

    # Regression using the other variable
    regression_pred = predict_with_regression(
        regression_model,
        test_data.loc[missing_mask],
        predictors=[predictor]
    )

    interpolation_mae, interpolation_rmse = calculate_errors(
        actual,
        interpolation_pred
    )

    regression_mae, regression_rmse = calculate_errors(
        actual,
        regression_pred
    )

    return {
        "block": block_id,
        "target": target,
        "predictor": predictor,
        "gap_size": gap_size,
        "start_idx": start_idx,
        "interpolation_mae": interpolation_mae,
        "interpolation_rmse": interpolation_rmse,
        "regression_mae": regression_mae,
        "regression_rmse": regression_rmse
    }


def validate_cross_variable_methods(
    data,
    block_ids,
    gap_sizes,
    positions_per_gap=5,
    edge_buffer=5
):
    """
    Validate humidity-pressure reconstruction across multiple
    blocks, gap sizes and gap locations.
    """

    results = []

    scenarios = [
        {
            "target": "humidity",
            "predictor": "pressure"
        },
        {
            "target": "pressure",
            "predictor": "humidity"
        }
    ]

    for scenario in scenarios:

        target = scenario["target"]
        predictor = scenario["predictor"]

        for block_id in block_ids:

            block_data = (
                data[data["block"] == block_id]
                .reset_index(drop=True)
            )

            for gap_size in gap_sizes:

                min_start = edge_buffer

                max_start = (
                    len(block_data)
                    - gap_size
                    - edge_buffer
                )

                if max_start < min_start:
                    continue

                positions = np.unique(
                    np.linspace(
                        min_start,
                        max_start,
                        positions_per_gap,
                        dtype=int
                    )
                )

                for start_idx in positions:

                    result = evaluate_cross_variable_reconstruction(
                        data,
                        block_id=block_id,
                        target=target,
                        predictor=predictor,
                        gap_size=gap_size,
                        start_idx=start_idx
                    )

                    results.append(result)

    return pd.DataFrame(results)