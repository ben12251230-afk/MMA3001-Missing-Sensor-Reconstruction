import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def create_missing_gap(data, gap_size, start_idx):
    """
    Create an artificial missing-data gap in a temperature time series.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing 'time' and 'temperature'.

    gap_size : int
        Number of consecutive measurements to remove.

    start_idx : int
        Index where the artificial gap begins.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the original values in 'actual'
        and missing values in 'temperature'.
    """
    experiment = (
        data[["time", "temperature"]]
        .copy()
        .reset_index(drop=True)
    )

    if gap_size <= 0:
        raise ValueError("gap_size must be greater than zero.")

    if start_idx < 0 or start_idx + gap_size > len(experiment):
        raise ValueError("Missing gap is outside the available data range.")

    experiment["actual"] = experiment["temperature"]

    missing_idx = list(
        range(start_idx, start_idx + gap_size)
    )

    experiment.loc[
        missing_idx,
        "temperature"
    ] = np.nan

    return experiment


def linear_interpolation(data):
    """
    Reconstruct missing temperature values using linear interpolation.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing 'time' and 'temperature'.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with a 'linear' column containing
        reconstructed temperature values.
    """
    result = data.copy()

    interpolated = (
        result
        .set_index("time")["temperature"]
        .interpolate(method="time")
    )

    result["linear"] = interpolated.to_numpy()

    return result


def cubic_spline_interpolation(data):
    """
    Reconstruct missing temperature values using cubic spline interpolation.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing 'time' and 'temperature'.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with a 'cubic' column containing
        reconstructed temperature values.
    """
    result = data.copy()

    x = (
        result["time"] - result["time"].min()
    ).dt.total_seconds().to_numpy() / 60

    known_mask = result["temperature"].notna().to_numpy()
    missing_mask = ~known_mask

    if known_mask.sum() < 2:
        raise ValueError(
            "Not enough known measurements for cubic spline interpolation."
        )

    spline = CubicSpline(
        x[known_mask],
        result.loc[
            known_mask,
            "temperature"
        ].to_numpy()
    )

    result["cubic"] = result["temperature"]

    result.loc[
        missing_mask,
        "cubic"
    ] = spline(x[missing_mask])

    return result