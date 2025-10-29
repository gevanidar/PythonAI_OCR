from loader.data_loader import load_data_from_root_folder
from modelling.model_helper import (
    setup_train_and_test_data,
    setup_decision_tree_classifier,
    get_default_model_name,
    setup_models,
    print_report,
)
from sklearn import metrics
import pandas as pd
import numpy as np
import time

from enum import Enum

DISPLAY_DATA_INFO = False


class DataSet(Enum):
    """
    The different folders to be used as root folder when running the project.
    Used to create easier iteration steps between testing different models.

    """

    SMALLER = "data/letters_smaller/"
    SMALL = "data/letters_small/"
    MEDIUM = "data/letters_medium/"
    FULL = "data/letters/"


def test_min_samples(splits, leaves):
    """
    Helped function for testing over several 'splits' and 'leaves' as parameter to check performance.

    Args:
        splits (List(int)): containing min_samples_splits
        leaves (List(int)): containing min_samples_leaves
    """
    for split in splits:
        for leave in leaves:
            start = time.time()
            model = setup_decision_tree_classifier(
                X_train,
                y_train,
                min_samples_split=split,
                min_samples_leaves=leave,
            )
            training_end = time.time()

            total = training_end - start
            info = f"The training took a ({total}) seconds to run"
            predictions = model.predict(X_test)

            end = time.time()
            total = end - start
            info += f"\nThe prediction took a ({total}) seconds to run"

            model_params = [f"min_samples_{split=}", f"min_samples_{leave=}"]

            print_report(
                y_test, predictions, get_default_model_name(), info, model_params
            )


if __name__ == "__main__":
    # Change the below to a root folder path for the dataset
    # Examples:
    # 'data/root_folder'
    # 'root_folder'
    root_folder = DataSet.FULL.value

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

    splits = [2, 4]
    leaves = [1, 2, 4, 8]
    test_min_samples(splits, leaves)
    # model = setup_decision_tree_classifier( X_train, y_train, min_samples_split=min_samples_split, min_samples_leaves=min_samples_leaves, )

    # predictions = model.predict(X_test)

    # end = time.time()
    # total = end - start

    # model_params = [f"{min_samples_split=}", f"{min_samples_leaves=}"]

    # print_report(y_test, predictions, get_default_model_name(), total, model_params)


# Deprecated
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
    print_report(y_test, predictions)
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
