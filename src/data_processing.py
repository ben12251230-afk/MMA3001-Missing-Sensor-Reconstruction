import json
import pandas as pd


def extract_temperature(json_string):
    """
    Extract the timestamp and temperature value from one JSON sensor record.

    Parameters
    ----------
    json_string : str
        JSON-formatted sensor measurement data.

    Returns
    -------
    tuple
        Timestamp and temperature value. Returns (None, None)
        if no temperature measurement is found.
    """
    data = json.loads(json_string)

    for item in data:
        if item["variable"]["name"] == "Temperature":
            values = item.get("values", [])

            if values:
                timestamp = values[0]["time"]
                temperature = values[0]["value"]

                return timestamp, temperature

    return None, None
def extract_environmental_variables(json_string):
    """
    Extract temperature, humidity, pressure and timestamp
    from an environmental sensor JSON record.

    Parameters
    ----------
    json_string : str
        JSON-formatted environmental sensor record.

    Returns
    -------
    pandas.Series
        Series containing timestamp, temperature,
        humidity and pressure.
    """
    data = json.loads(json_string)

    values = {
        "timestamp": None,
        "temperature": None,
        "humidity": None,
        "pressure": None
    }

    target_variables = {
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Pressure": "pressure"
    }

    for item in data:

        name = item.get(
            "variable",
            {}
        ).get("name")

        if name not in target_variables:
            continue

        item_values = item.get(
            "values",
            []
        )

        if len(item_values) == 0:
            continue

        record = item_values[0]

        timestamp = record.get("time")
        value = record.get("value")

        if value is None:
            continue

        if (
            values["timestamp"] is None
            and timestamp is not None
        ):
            values["timestamp"] = timestamp

        values[
            target_variables[name]
        ] = value

    return pd.Series(values)

def prepare_temperature_data(environmental_data, sensor_id):
    """
    Prepare a clean temperature time series for one sensor.

    The function selects one sensor, extracts temperature measurements
    from the JSON data, converts timestamps to datetime, removes missing
    values, checks duplicate timestamps, and returns cleaned data.

    Parameters
    ----------
    environmental_data : pandas.DataFrame
        Raw environmental sensor dataset containing
        'sensorid' and 'jsondata' columns.
        The 'jsondata' column contains JSON-formatted
        environmental measurements.

    sensor_id : int
        ID of the sensor to analyse.

    Returns
    -------
    pandas.DataFrame
        Clean dataframe containing time and temperature columns.

    Raises
    ------
    ValueError
        If the selected sensor does not exist or duplicate timestamps
        contain conflicting temperature values.
    """

    sensor = environmental_data[
        environmental_data["sensorid"] == sensor_id
    ].copy()

    if sensor.empty:
        raise ValueError(
            f"Sensor ID {sensor_id} was not found in the dataset."
        )

    sensor[["timestamp", "temperature"]] = sensor["jsondata"].apply(
        lambda x: pd.Series(extract_temperature(x))
    )

    sensor["time"] = pd.to_datetime(
        sensor["timestamp"],
        unit="s"
    )

    temp_data = (
        sensor[["time", "temperature"]]
        .dropna()
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Check whether duplicate timestamps disagree
    duplicate_check = (
        temp_data.groupby("time")["temperature"]
        .nunique()
    )

    conflicts = duplicate_check[duplicate_check > 1]

    if len(conflicts) > 0:
        raise ValueError(
            "Conflicting temperature values were found "
            "at duplicate timestamps."
        )

    # Safe to remove identical duplicates
    temp_clean = (
        temp_data
        .drop_duplicates(subset="time", keep="first")
        .sort_values("time")
        .reset_index(drop=True)
        .copy()
    )

    return temp_clean


def add_time_gap(data):
    """
    Calculate the time difference between consecutive measurements.

    Parameters
    ----------
    data : pandas.DataFrame
        Temperature dataframe containing a 'time' column.

    Returns
    -------
    pandas.DataFrame
        Copy of the dataframe with an additional 'time_gap' column.
    """

    result = data.copy()

    result["time_gap"] = result["time"].diff()

    return result


def identify_continuous_blocks(
    data,
    expected_minutes=10,
    tolerance_seconds=5
):
    """
    Identify continuous measurement blocks based on sampling interval.

    A measurement is treated as continuous when the time gap is within
    the specified tolerance of the expected sampling interval.

    Parameters
    ----------
    data : pandas.DataFrame
        Clean temperature dataframe containing time and time_gap.

    expected_minutes : float, default=10
        Expected sensor sampling interval in minutes.

    tolerance_seconds : float, default=5
        Allowed deviation from the expected interval.

    Returns
    -------
    pandas.DataFrame
        Dataframe with regular_gap, new_block, and block columns.
    """

    result = data.copy()

    expected_gap = pd.Timedelta(minutes=expected_minutes)
    tolerance = pd.Timedelta(seconds=tolerance_seconds)

    result["regular_gap"] = (
        (result["time_gap"] - expected_gap).abs()
        <= tolerance
    )

    # First measurement starts the first block
    result.loc[result.index[0], "regular_gap"] = False

    result["new_block"] = ~result["regular_gap"]

    result["block"] = result["new_block"].cumsum()

    return result


def summarise_blocks(data):
    """
    Summarise each continuous temperature block.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing a 'block' column.

    Returns
    -------
    pandas.DataFrame
        Summary showing start time, end time, number of measurements,
        and duration for each block.
    """

    blocks = data.groupby("block").agg(
        start=("time", "min"),
        end=("time", "max"),
        measurements=("time", "size")
    )

    blocks["duration"] = blocks["end"] - blocks["start"]

    return blocks.sort_values(
        "measurements",
        ascending=False
    )