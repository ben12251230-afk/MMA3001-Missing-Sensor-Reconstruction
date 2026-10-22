
import numpy as np
import pandas as pd


def create_missing_windows(
    validation_data,
    gap_durations=(10, 30, 60, 120),
    n_tests=15
):
    """
    Create artificial missing-temperature windows.

    Parameters
    ----------
    validation_data : pd.DataFrame
        Time-series measurements with a time column.
    gap_durations : tuple
        Missing-window durations in minutes.
    n_tests : int
        Number of test positions per duration.

    Returns
    -------
    pd.DataFrame
        Test windows containing duration, test number,
        start time, end time and number of hidden readings.
    """
    # Validate input parameters
    if "time" not in validation_data.columns:
        raise ValueError("Input data must contain a 'time' column.")

    if not pd.api.types.is_datetime64_any_dtype(validation_data["time"]):
        raise TypeError("The 'time' column must contain datetime values.")

    if validation_data["time"].isna().any():
        raise ValueError("The 'time' column contains missing values.")

    if not isinstance(n_tests, (int, np.integer)) or n_tests <= 0:
        raise ValueError("n_tests must be a positive integer.")

    if not gap_durations or any(
    not isinstance(d, (int, np.integer)) or d <= 0
    for d in gap_durations
):
        raise ValueError("gap_durations must contain positive integers.")
    
    data = (
        validation_data
        .sort_values("time")
        .reset_index(drop=True)
    )

    if len(data) < 25:
        raise ValueError("Not enough data to create test windows.")

    missing_windows = []

    for duration in gap_durations:

        start_positions = np.linspace(
            4,
            len(data) - 20,
            n_tests,
            dtype=int
        )

        for test_number, start_idx in enumerate(start_positions, 1):

            start_time = data.loc[start_idx, "time"]
            end_time = start_time + pd.Timedelta(minutes=duration)
            cutoff_time = end_time - pd.Timedelta(seconds=5)

            hidden = data[
                (data["time"] >= start_time) &
                (data["time"] < cutoff_time)
            ]

            before = data[data["time"] < start_time].tail(4)
            after = data[data["time"] >= cutoff_time].head(4)

            if len(hidden) == 0 or len(before) < 4 or len(after) < 4:
                continue

            missing_windows.append({
                "duration_min": duration,
                "test_number": test_number,
                "start_time": start_time,
                "end_time": end_time,
                "n_hidden": len(hidden)
            })

    return pd.DataFrame(missing_windows)


from scipy.interpolate import CubicSpline

def endpoint_constrained_combined_regression(
    before,
    hidden,
    after,
    combined_model
):
    """
    Reconstruct missing temperatures using combined regression
    with constraints at both known temperature endpoints.
    """

    # 1. Known temperature measurements at both boundaries
    left = before.iloc[-1]
    right = after.iloc[0]

    t0 = left["time"]
    t1 = right["time"]

    T0 = left["temperature"]
    T1 = right["temperature"]

    # 2. Original regression predictions inside the gap
    X_hidden = hidden[["humidity", "pressure"]]

    raw_predictions = combined_model.predict(X_hidden)

    # 3. Original regression predictions at both endpoints
    X_endpoints = pd.DataFrame({
        "humidity": [left["humidity"], right["humidity"]],
        "pressure": [left["pressure"], right["pressure"]]
    })

    endpoint_predictions = combined_model.predict(X_endpoints)

    pred_left = endpoint_predictions[0]
    pred_right = endpoint_predictions[1]

    # 4. Relative time position inside the missing window
    alpha = (
        (hidden["time"] - t0) / (t1 - t0)
    ).to_numpy(dtype=float)

    # 5. Correct regression predictions to match both endpoints
    constrained_predictions = (
        raw_predictions
        + (1 - alpha) * (T0 - pred_left)
        + alpha * (T1 - pred_right)
    )

    return constrained_predictions


def validate_reconstruction(
    validation_data,
    missing_windows,
    humidity_model,
    pressure_model,
    combined_model,
    temperature_range
):
    """
    Evaluate seven temperature reconstruction methods using
    the same artificially missing temperature windows.

    Returns
    -------
    pd.DataFrame
        MAE, RMSE and NRMSE for each method and test window.
    """

    results = []

    for _, window in missing_windows.iterrows():

        start_time = window["start_time"]
        end_time = window["end_time"]

        cutoff_time = end_time - pd.Timedelta(seconds=5)

        # Measurements withheld for validation
        hidden = validation_data[
            (validation_data["time"] >= start_time) &
            (validation_data["time"] < cutoff_time)
        ].copy()

        # Known temperature measurements outside the gap
        before = validation_data[
            validation_data["time"] < start_time
        ].tail(4)

        after = validation_data[
            validation_data["time"] >= cutoff_time
        ].head(4)

        if len(hidden) == 0 or len(before) < 4 or len(after) < 4:
            continue

        # Endpoint-Constrained Combined Regression
        constrained_pred = endpoint_constrained_combined_regression(
            before=before,
            hidden=hidden,
            after=after,
            combined_model=combined_model
        )

        # Ground truth: only used for evaluation
        y_true = hidden["temperature"].to_numpy(dtype=float)

        # 1. Persistence baseline
        persistence_pred = np.full(
            len(hidden),
            before["temperature"].iloc[-1]
        )

        # Prepare known points for interpolation
        known = pd.concat([before, after]).sort_values("time")

        x_known = (
            known["time"] - start_time
        ).dt.total_seconds().to_numpy(dtype=float)

        x_hidden = (
            hidden["time"] - start_time
        ).dt.total_seconds().to_numpy(dtype=float)

        y_known = known["temperature"].to_numpy(dtype=float)

        # 2. Linear interpolation
        linear_pred = np.interp(
            x_hidden,
            x_known,
            y_known
        )

        # 3. Cubic spline interpolation
        spline = CubicSpline(
            x_known,
            y_known,
            bc_type="natural"
        )

        cubic_pred = spline(x_hidden)

        # 4. Humidity-only regression
        humidity_pred = humidity_model.predict(
            hidden[["humidity"]]
        )

        # 5. Pressure-only regression
        pressure_pred = pressure_model.predict(
            hidden[["pressure"]]
        )

        # 6. Combined regression
        combined_pred = combined_model.predict(
            hidden[["humidity", "pressure"]]
        )

        predictions = {
            "Persistence": persistence_pred,
            "Linear": linear_pred,
            "Cubic spline": cubic_pred,
            "Humidity regression": humidity_pred,
            "Pressure regression": pressure_pred,
            "Combined regression": combined_pred,
            "Endpoint-constrained regression": constrained_pred
        }

        # Calculate reconstruction errors
        for method_name, y_pred in predictions.items():

            mae = np.mean(
                np.abs(y_true - y_pred)
            )

            rmse = np.sqrt(
                np.mean((y_true - y_pred) ** 2)
            )

            nrmse = (
                rmse / temperature_range * 100
                if temperature_range > 0
                else np.nan
            )

            results.append({
                "duration_min": window["duration_min"],
                "test_number": window["test_number"],
                "start_time": start_time,
                "n_hidden": len(hidden),
                "method": method_name,
                "MAE": mae,
                "RMSE": rmse,
                "NRMSE (%)": nrmse
            })

    return pd.DataFrame(results)

