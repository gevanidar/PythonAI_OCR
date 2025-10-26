# 20251026
# SVC -> SDGClassifier
Tried SVC and ended up with a very low accuracy and test result. I read up that SGDCLassifier would be better.
'The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples and may be impractical beyond tens of thousands of samples. For large datasets consider using LinearSVC or SGDClassifier instead, possibly after a Nystroem transformer or other Kernel Approximation.'
- Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
- Reference: https://sklearn.vercel.app/docs/classes/SVC

# Changing approach to Decision Tree
Since SVC and SDGClassifier are used for linear regression it might not be optimal for OCR.
I have changed the approach to using a Decision Tree
