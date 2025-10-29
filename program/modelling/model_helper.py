from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
)
import numpy as np


def print_report(y_test, predictions, model_name):
    """
    Create accuracy and classification report from the test predictions.
    Prints the absolute error, mean squared error and roo mean squared error for the prediction
    The size of y_test and predictions are equal.
    Element i in y_test are assumed to correspond to the element i in predictions

    Args:
        y_test (List(str)): The correct labels
        predictions (List(str)): The predictions done by the model
    """
    SEPARATOR = "-----------------------------------------------------------------------------------------"

    # Compare the prediction from the X_test dataset with the correct labels
    accuracy = accuracy_score(y_test, predictions)

    # Create a classification report of y_test compared to the predictions
    report = classification_report(y_test, predictions)

    # Print the dataframe
    print(SEPARATOR)
    df = pd.DataFrame({"Actual Value": y_test, "Predicted Values": predictions})
    print(df)
    # Print the result of he report and accuracy
    print(SEPARATOR)
    print(f"Accuracry: {accuracy:0.5f}\n")
    print(SEPARATOR)
    print("Classification report")
    print(report)
    print(SEPARATOR)
    print(
        f"Mean Absolute Error ({model_name}):",
        mean_absolute_error(y_test, predictions),
    )
    print(
        f"Mean Squared Error ({model_name}):",
        mean_squared_error(y_test, predictions),
    )
    print(
        f"Root Mean Squared Error ({model_name}):",
        np.sqrt(mean_squared_error(y_test, predictions)),
    )
    print(SEPARATOR)


def setup_train_and_test_data(X, y):
    """
    Setup the traning and test data with controlled parameters
    The data is split into Training and Test data sets;

    Args:
        X (List): The data
        y (List): Labels for the data

    Returns:
        X_train (List(type of X)): The Training data
        X_test (List(type of X)): The Test data
        y_train (List(type of y)): The Training labels
        y_test (List(type of y)): The Test labels
    """
    test_size = 0.2  # [X,y]_test size of the orginal [X,y] data set
    random_state = 50  # Replicate result (Used to compare results between changes)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_default_model():
    return DecisionTreeClassifier


def get_default_model_name():
    return get_default_model().__name__


def setup_default_model(X_train, y_train):
    return setup_model(X_train, y_train, DecisionTreeClassifier)


def setup_model(X_train, y_train, model_class):
    """
    Setup the model and fit it to the data

    Args:
        X_train (List): The training data used for fitting the model
        y_train (List): The training labels used for fitting the model

    Returns:
        model (DecisionTreeClassifier): DecisionTreeClassifier model fit to data
    """
    # Create the model and fit it to the training data
    model = model_class()
    model.fit(X_train, y_train)

    return model


def setup_models(X_train, y_train):
    """
    Setup the model and fit it to the data

    Args:
        X_train (List): The training data used for fitting the model
        y_train (List): The training labels used for fitting the model

    Returns:
        dtc (DecisionTreeClassifier): DecisionTreeClassifier model fit to data
        dtr (DecisionTreeRegressor): DecisionTreeRegressor model fit to data
        br (BaggingRegressor): BaggingRegressor model fit to data
    """
    # Create the model and fit it to the training data
    dtc = setup_model(X_train, y_train, DecisionTreeClassifier)
    dtr = setup_model(X_train, y_train, DecisionTreeRegressor)

    br = BaggingRegressor(
        dtr, n_estimators=100, max_features=3, max_samples=0.5
    )  # get Boostrap aggregation ensemble regressor
    br.fit(X_train, y_train)

    return dtc, dtr, br
