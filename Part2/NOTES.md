# 20251026
# SVC -> SDGClassifier
Tried SVC and ended up with a very low accuracy and test result. I read up that SGDCLassifier would be better.
'The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples. For large datasets consider using LinearSVC or SGDClassifier instead, possibly after a Nystroem transformer or other Kernel Approximation.'
- Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
- Reference: https://sklearn.vercel.app/docs/classes/SVC

# Changing approach to Decision Tree
Since SVC and SDGClassifier are used for linear regression it might not be optimal for OCR.
I have changed the approach to using a Decision Tree

# After testing Decision tree
After testing parameters
- min_samples_split
- min_samples_leaves
There was a tendancy that increasing the parameter caused higher accuracy for borth split and leaves.

The Accuracy is around 82%. I believe that other parameters might also change the result.
However I believe that the only wa t increase the accuracy now is to remove outliers.


# Further testing with pixel size
Pixel size 32x32 gives an Accuracy around 70% while reducing the number of pixels to 16x16 gives almost 75% while continuing down to 8x8 pixels gives a little higher. (Medium sized test sample)
