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
Accuracry: 0.68389

-----------------------------------------------------------------------------------------
Classification report
              precision    recall  f1-score   support

           A       0.57      0.38      0.46      3426
           B       0.52      0.72      0.61      1743
           C       0.50      0.86      0.63      2701
           D       0.58      0.71      0.64      3166
           E       0.83      0.74      0.78      6738
           F       0.79      0.64      0.71      2257
           G       0.19      0.50      0.28      1064
           H       0.59      0.69      0.63      2422
           I       0.59      0.49      0.53      2818
           J       0.47      0.74      0.57       861
           K       0.90      0.14      0.24       869
           L       0.58      0.67      0.62      4378
           M       0.89      0.77      0.82      2407
           N       0.72      0.69      0.71      4204
           O       0.89      0.85      0.87     13031
           P       0.87      0.64      0.74      2168
           Q       0.50      0.19      0.27       898
           R       0.62      0.68      0.65      4051
           S       0.86      0.87      0.87      5266
           T       0.91      0.38      0.54      5980
           U       0.82      0.63      0.71      3276
           V       0.66      0.72      0.69      1507
           W       0.59      0.87      0.70      1496
           X       0.64      0.65      0.64      1032
           Y       0.50      0.75      0.60      1364
           Z       0.34      0.82      0.49       999

    accuracy                           0.68     80122
   macro avg       0.65      0.65      0.62     80122
weighted avg       0.73      0.68      0.69     80122

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
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


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
    model = SGDClassifier()
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
