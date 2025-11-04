# Project 1

## Questions
- Is the data complete?
  - Yes, however 0 and O share the same data. The focus will be on Letters, so that they share the data will not be a problem. 
- Are there any null-values?
  - The amount of data per label is not the same, for some letters there are 1000 images and for some around 2000. They are however all labeled, so the data itself does not contain null values.
- Are there any outliers?
  - Most likely, since it is hand written text. The data is too large, so I will have to clean the data from the outliers somehow.
    - How?
- What are the datatypes of the data?
  - JPG files are the input. So row and columns for the input, along with the label -> [Label, Image([X,Y])]
- What fields in your data do you want to use?
  - All Pixels in the images?
- How can you convert all fields to a numerical value?
  - The images are already numerical values since they are pixels. The labels could be converted to Numeric vaulues, however that is probably not needed.


## Type of problem
- Classification, Supervised learning?
The problem selectd is to use machine learning to classify (Classification) characters.
Datasets that have been looked at contain 1 character per image and thus belong to one Classification each.
Does the data contain labels or do I need to create it myself? Depending on data set size. - Or use unsupervised learning? How to apply it to Character Recognition?

## Describe the data found
First select the data and then look at the data using python, (see lecture notes).

- [ ] TODO: 

### Amount of data?
- A-Z, total amount of images: 37889. There are more images, I will start of with this smaller dataset and add data depending on how long it takes. Perhaps do a run with fewer data points first.
- After testing I will start with a lot smaller set to first work out how to get it to work and then setup for larget datasets once the model can train and test itself.
### Quality of the data
- The images ar 32x32 pixels large. 

## Reference:
Dataset: [Kaggle - vaibhao - handwritten characters](https://www.kaggle.com/datasets/vaibhao/handwritten-characters)

## Selected project
Hand written text
