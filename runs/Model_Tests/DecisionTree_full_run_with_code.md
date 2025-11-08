[[1 1 1 ... 2 1 0]
 [0 0 0 ... 0 1 1]
 [0 0 0 ... 1 1 1]
 ...
 [0 0 0 ... 0 0 0]
 [1 1 1 ... 2 0 0]
 [1 0 1 ... 1 1 0]] ['A' 'A' 'A' ... 'Z' 'Z' 'Z']
Splitting data into Training set (X_train, y_train) and a Trsting set (X_test, y_test)
Creating the model and fitting it to the model
Make a prediction on the test data X_test
Score the prediction according to the y_test
Create a classificaton report from the predictions
-----------------------------------------------------------------------------------------
Accuracry: 0.82238

-----------------------------------------------------------------------------------------
Classification report
              precision    recall  f1-score   support

           A       0.73      0.73      0.73      3426
           B       0.75      0.73      0.74      1743
           C       0.85      0.85      0.85      2701
           D       0.80      0.80      0.80      3166
           E       0.88      0.88      0.88      6738
           F       0.77      0.77      0.77      2257
           G       0.51      0.50      0.51      1064
           H       0.77      0.76      0.77      2422
           I       0.56      0.58      0.57      2818
           J       0.74      0.75      0.74       861
           K       0.70      0.68      0.69       869
           L       0.69      0.71      0.70      4378
           M       0.87      0.84      0.85      2407
           N       0.82      0.85      0.83      4204
           O       0.94      0.94      0.94     13031
           P       0.85      0.83      0.84      2168
           Q       0.49      0.49      0.49       898
           R       0.83      0.83      0.83      4051
           S       0.91      0.91      0.91      5266
           T       0.87      0.87      0.87      5980
           U       0.85      0.86      0.86      3276
           V       0.80      0.80      0.80      1507
           W       0.84      0.80      0.82      1496
           X       0.78      0.74      0.76      1032
           Y       0.76      0.76      0.76      1364
           Z       0.77      0.74      0.76       999

    accuracy                           0.82     80122
   macro avg       0.77      0.77      0.77     80122
weighted avg       0.82      0.82      0.82     80122

-----------------------------------------------------------------------------------------
```python3
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
```
