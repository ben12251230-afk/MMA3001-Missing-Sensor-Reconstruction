"""Machine learning models for environmental sensor reconstruction."""

import pandas as pd
from sklearn.linear_model import LinearRegression


def train_linear_regression(data, predictors, target):
    """
    Train a linear regression model using environmental sensor data.

    Parameters
    ----------
    data : pandas.DataFrame
        Training dataset containing predictor and target columns.

    predictors : list of str
        Predictor column names, such as 'humidity' and 'pressure'.

    target : str
        Target variable to predict, such as 'temperature'.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        Fitted linear regression model.

    Raises
    ------
    KeyError
        If required predictor or target columns are missing.

    ValueError
        If no valid training observations remain or the
        training data cannot be fitted.

    Notes
    -----
    Rows containing missing predictor or target values are
    removed before training.

    The function uses ordinary least squares regression.
    Training and testing data must be separated before
    calling this function to avoid data leakage.
    """
    clean_data = data.dropna(
        subset=predictors + [target]
    )

    X = clean_data[predictors]
    y = clean_data[target]

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_with_regression(model, data, predictors):
    """
    Generate predictions using a fitted linear regression model.

    Parameters
    ----------
    model : sklearn.linear_model.LinearRegression
        Previously fitted regression model.

    data : pandas.DataFrame
        Input measurements containing the predictor columns.

    predictors : list of str
        Predictor column names in the same order as those
        used during model training.

    Returns
    -------
    numpy.ndarray
        Predicted values, with one prediction per input row.

    Raises
    ------
    KeyError
        If required predictor columns are missing.

    ValueError
        If predictor values are invalid or incompatible
        with the fitted regression model.

    sklearn.exceptions.NotFittedError
        If the regression model has not been fitted.

    Notes
    -----
    The model must be trained before making predictions.

    This function does not automatically remove missing
    predictor values. Valid numerical inputs are required.

    For this project, the regression models are used to
    predict temperature in degrees Celsius.
    """
    X = data[predictors]

    return model.predict(X)