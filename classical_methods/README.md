# Classical Methods
This section is the easier (but still powerful) part of classification where we will be using scikit-learn to implement `Random Forest` and `Logistical Regression` classifiers. Notice how few lines are needed in the `main.py` to train and evaluate a model on a binary classification task. While recommended to, the reader is not required to read up on or understand the models but simply instantiate them with e.g. `RandomForestClassifier()`. 
In general most sklearn classes does not require many (if at all) arguments but they should often be experimented with to boost performance if near an adequate solution.

## Background

### Random Forest
A decision tree learns a series of decision rules based on individual features where the most important feature is split first until a decision is reached or the maximum depth (max feature decisions) is reached. 

An example could be deciding if a student will pass or fail an exam based on the feature variables time spent studying per week, number of attended lectures and height and with a maximum allowed depth of 2 nodes. The model might find that time spent studying is the most important feature, attended lectures second most and that height should not be considered since 2 was the max feature decisions allowed. The tree might learn that less than 2.5 hours studying a week is enough to predict a fail. If the student studied more than 2.5 hours the model finds it difficult without splitting success and failure based on if number of attended lectures is more or less than 5.

An example decision tree is shown below.
![loss](decision_tree.jpg)

A random forest, as the name implies, is built by averaging many decision trees on different subsamples of the dataset to control overfitting and increase the models predictive power. Some of the most important hyperparameters to decide when tuning the model are maximum three depth, number of trees and minimum samples per leaf:
```python
sklearn.ensemble.RandomForestClassifier(max_depth=4)
sklearn.ensemble.RandomForestClassifier(n_estimators=100)
sklearn.ensemble.RandomForestClassifier(min_samples_leaf=1)
```

### Logistical Regression

## Prerequisites
- scikit-learn 1.1.1

## Usage
Run the following in terminal or conda prompt to train RF/LogReg models and output accuracy:
```
python main.py
```

