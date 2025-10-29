from loader.data_loader import load_data_from_root_folder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


DISPLAY_DATA_INFO = True

SEPARATOR = "-----------------------------------------------------------------------------------------"


def print_report(y_test, predictions):
    """
    Create accuracy and classification report from the test predictions.
    The size of y_test and predictions are equal.
    Element i in y_test are assumed to correspond to the element i in predictions

    Args:
        y_test (List(str)): The correct labels
        predictions (List(str)): The predictions done by the model
    """
    # Compare the prediction from the X_test dataset with the correct labels
    accuracy = accuracy_score(y_test, predictions)

    # Create a classification report of y_test compared to the predictions
    report = classification_report(y_test, predictions)

    # Print the result of he report and accuracy
    print(SEPARATOR)
    print(f"Accuracry: {accuracy:0.5f}\n")
    print(SEPARATOR)
    print("Classification report")
    print(report)
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


def setup_model(X_train, y_train):
    """
    Setup the model and fit it to the data

    Args:
        X_train (List): The training data used for fitting the model
        y_train (List): The training labels used for fitting the model
    """
    # Create the model and fit it to the training data
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return model


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
