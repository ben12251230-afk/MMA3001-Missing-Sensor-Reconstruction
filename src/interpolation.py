import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def create_missing_gap(data, gap_size, start_idx):
    """Create an artificial gap by hiding consecutive temperature measurements.

    The original temperatures are retained in an 'actual' column as
    ground truth, while the selected 'temperature' entries become NaN.

    Parameters
    ----------
    data : pandas.DataFrame
        Input measurements containing 'time' and 'temperature'.
    gap_size : int
        Number of consecutive observations to hide (not elapsed minutes).
    start_idx : int
        Zero-based index of the first observation to hide.

    Returns
    -------
    pandas.DataFrame
        Copy containing 'time', 'temperature', and 'actual' columns.

    Raises
    ------
    ValueError
        If gap_size is not positive or the requested gap lies outside
        the available data range.

    Notes
    -----
    This function was used in early exploratory experiments. The final
    seven-method validation uses timestamp-based windows created by
    'create_missing_windows' in the validation module.
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
    """Reconstruct missing temperature values using time-based linear interpolation.

    Parameters
    ----------
    data : pandas.DataFrame
        Measurements with a datetime 'time' column and a numerical
        'temperature' column containing NaN at missing observations.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with a 'linear' column containing interpolated
        temperatures; the original 'temperature' column is unchanged.

    Notes
    -----
    Pandas 'interpolate(method="time")' uses actual timestamp spacing.
    Interior missing values are estimated between known observations;
    missing values at the beginning may remain unfilled. The function
    assumes valid datetime values and does not explicitly validate them.
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
    """Reconstruct missing temperature values using cubic spline interpolation.

    Parameters
    ----------
    data : pandas.DataFrame
        Measurements with datetime 'time' and numerical 'temperature'
        columns; missing temperatures are represented by NaN.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with a 'cubic' column containing spline
        estimates at the missing timestamps.

    Raises
    ------
    ValueError
        If fewer than two known temperature measurements are available.

    Notes
    -----
    This exploratory helper uses SciPy's CubicSpline with its default
    'not-a-knot' boundary conditions. The final seven-method validation
    separately uses natural boundary conditions on four known readings
    before and four after each missing window. Cubic splines can
    overshoot and produce unrealistic intermediate values.
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