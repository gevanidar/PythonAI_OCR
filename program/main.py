from loader.data_loader import load_data_from_root_folder
from modelling.model_helper import setup_train_and_test_data, setup_model, print_report
import pandas as pd

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

    model = setup_model(X_train, y_train)

    # Create a prediction on the X_test dataset using the model.
    predictions = model.predict(X_test)

    print_report(y_test, predictions)
