"""Machine learning models for environmental sensor reconstruction."""

import pandas as pd
from sklearn.linear_model import LinearRegression


def train_linear_regression(data, predictors, target):
    """
    Train a multiple linear regression model.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing predictor and target variables.

    predictors : list of str
        Column names used as predictor variables.

    target : str
        Column name of the variable to predict.

    Returns
    -------
    sklearn.linear_model.LinearRegression
        Trained linear regression model.
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
    Predict values using a trained regression model.

    Parameters
    ----------
    model : sklearn.linear_model.LinearRegression
        Previously trained regression model.

    data : pandas.DataFrame
        Data containing predictor variables.

    predictors : list of str
        Predictor columns used by the model.

    Returns
    -------
    numpy.ndarray
        Predicted values.
    """
    X = data[predictors]

    return model.predict(X)