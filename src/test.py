from loader.data_loader import load_data_from_root_folder
from modelling.model_helper import (
    setup_train_and_test_data,
    setup_default_model,
    setup_models,
    print_report,
)
from sklearn import metrics
import pandas as pd
import numpy as np

from enum import Enum

DISPLAY_DATA_INFO = True


class DataSet(Enum):
    """
    The different folders to be used as root folder when running the project.
    Used to create easier iteration steps between testing different models.

    """

    SMALLER = "data/letters_smaller/"
    SMALL = "data/letters_small/"
    MEDIUM = "data/letters_medium/"
    FULL = "data/letters/"


if __name__ == "__main__":
    # Change the below to a root folder path for the dataset
    # Examples:
    # 'data/root_folder'
    # 'root_folder'
    root_folder = DataSet.SMALL.value

    X, y = load_data_from_root_folder(root_folder)

    if DISPLAY_DATA_INFO:
        # Display information about the data
        dataFrame = pd.DataFrame(X)
        dataFrame["label"] = y
        print(dataFrame.describe())
        print(dataFrame.values)
        print(type(X), type(y))
        print(X, y)

    X_train, X_test, y_train, y_test = setup_train_and_test_data(X, y)

    model = setup_default_model(X_train, y_train)

    predictions = model.predict(X_test)

    df = pd.DataFrame({"Actual Value": y_test, "Predicted Values": predictions})
    print(df)
    print(
        "Mean Absolute Error (DecisionTree):",
        metrics.mean_absolute_error(y_test, predictions),
    )
    print(
        "Mean Squared Error (DecisionTree):",
        metrics.mean_squared_error(y_test, predictions),
    )
    print(
        "Root Mean Squared Error (DecisionTree):",
        np.sqrt(metrics.mean_squared_error(y_test, predictions)),
    )
    print_report(y_test, predictions)


# Deprecated (Kept after run 20251029)
def test_multiple():
    btc, btr, br = setup_models(X_train, y_train)

    # Create a prediction on the X_test dataset using the model.
    btc_pred = btc.predict(X_test)
    btr_pred = btr.predict(X_test)
    br_pred = br.predict(X_test)

    df = pd.DataFrame(
        {
            "Actual Value": y_test,
            "Predicted Values": btr_pred,
            "Bagging Predicted Values": br_pred,
        }
    )
    print(df)
    print(
        "Mean Absolute Error (DecisionTree):",
        metrics.mean_absolute_error(y_test, btc_pred),
    )
    print(
        "Mean Squared Error (DecisionTree):",
        metrics.mean_squared_error(y_test, btc_pred),
    )
    print(
        "Root Mean Squared Error (DecisionTree):",
        np.sqrt(metrics.mean_squared_error(y_test, btc_pred)),
    )

    print(
        "Mean Absolute Error (DecisionTreeRegression):",
        metrics.mean_absolute_error(y_test, btr_pred),
    )
    print(
        "Mean Squared Error (DecisionTreeRegression):",
        metrics.mean_squared_error(y_test, btr_pred),
    )
    print(
        "Root Mean Squared Error (DecisionTreeRegression):",
        np.sqrt(metrics.mean_squared_error(y_test, btr_pred)),
    )

    print(
        "Mean Absolute Error (BaggingRegressor):",
        metrics.mean_absolute_error(y_test, br_pred),
    )
    print(
        "Mean Squared Error (BaggingRegressor):",
        metrics.mean_squared_error(y_test, br_pred),
    )
    print(
        "Root Mean Squared Error (BaggingRegressor):",
        np.sqrt(metrics.mean_squared_error(y_test, br_pred)),
    )
