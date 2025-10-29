from loader.data_loader import load_data_from_root_folder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


root_folder = "data/letters_medium/"
root_folder = "data/letters/"
root_folder = "data/letters_smaller/"
X, y = load_data_from_root_folder(root_folder)
print(X, y)


if __name__ == "__main__":
    print(
        "Splitting data into Training set (X_train, y_train) and a Trsting set (X_test, y_test)"
    )
    random_state = 50  # For replicatable result
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    print("Creating the model and fitting it to the model")
    # model = SVC(gamma=0.001)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    print("Make a prediction on the test data X_test")
    predictions = model.predict(X_test)
    print("Score the prediction according to the y_test")
    accuracy = accuracy_score(y_test, predictions)

    print("Create a classificaton report from the predictions")
    report = classification_report(y_test, predictions)

    separator = "-----------------------------------------------------------------------------------------"

    print(separator)
    print(f"Accuracry: {accuracy:0.5f}\n")
    print(separator)
    print("Classification report")
    print(report)
    print(separator)
