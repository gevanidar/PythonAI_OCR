# Python AI project
Using available methods in sklearn to fit a data model to hand written text.

# Part 1
See [[Part1.md]]

# Part 2
Runs available with code snippets, which are located in `runs/`.
  - Note: Runs from the SVC have not been stored, approximately 5% correct.

### Folder structure
`runs/` - contains informaton about test runs on the full dataset.
`program/` - contains the python project and the data.
`program/loader` - contains module for loading images.
`program/modelling` - contains module for modelling.

Latest code can be run inside `program/` with `python3 main.py` or `python main.py` depending on your system and installation.

## Requirements
### Data location
Data should be located in the folder inside `program/data/letters`, folder can be configured in `program/main.py` using te `root_folder` variable.

### Libraries
`python3`
`numpy`
`scikit-learn`
`pandas`

### Dataset
Training data used have been extracted from the `Train` Dataset.
Training data is a subset of the foldes in the `Train` folder, only using [A-Z] characters. Note: `0` has been used as `O`.
Reference: [vaibhao handwritten-characters](https://www.kaggle.com/datasets/vaibhao/handwritten-characters)


# Tested on
Linux

# By
Arvid Nilsson


# Refrences:
Documentation of scikit learn: [scikit-learn org](https://scikit-learn.org)
DataSet: [vaibhao handwritten-characters](https://www.kaggle.com/datasets/vaibhao/handwritten-characters)
Stack Overflow inspiration for improving algorithm: [Improve decision tree model](https://stackoverflow.com/questions/59147246/how-can-i-improve-the-accuracy-of-my-prediction-from-a-decision-tree-model-using)
