from loader.data_loader import load_data_from_root_folder
from modelling.model_helper import setup_train_and_test_data, setup_model, print_report
import pandas as pd

DISPLAY_DATA_INFO = True

if __name__ == "__main__":
    # TODO: Select root easier
    root_folder = "data/letters_medium/"
    root_folder = "data/letters/"
    root_folder = "data/letters_smaller/"

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
