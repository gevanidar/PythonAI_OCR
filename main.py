import os
import numpy as np
from PIL import Image


def load_images(root_folder):
    """
    Load images in subfolders of root_folder
    """

    images = []
    labels = []

    image_size = (16, 16)

    for image_folder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, image_folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.endswith((".jpg")):
                continue

            try:
                image_path = os.path.join(folder_path, filename)

                img = Image.open(image_path).convert("L")

                img = img.resize(image_size)

                img_array = np.array(img).flatten()

                images.append(img_array)
                labels.append(image_folder)

            except Exception as e:
                print(f"Error when processing the file {filename}: {e}")
    return np.array(images), np.array(labels)


from sklearn.model_selection import train_test_split

# Bad for OCR from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


root_folder = "data/letters_medium/"
root_folder = "data/letters/"
X, y = load_images(root_folder)
print(X, y)


# TODO: Change y labels so that they map to int (can be a simple index)
# def label_to_value()
# def value_to_label()


start = True
if start:
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
