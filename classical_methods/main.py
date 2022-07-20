# -----------------------------------------------------------
# classical methods for binary classification in sklearn
#
# Torjus Nilsen, Kongsberg, Norway
# email tornil1996@hotmail.com
# -----------------------------------------------------------

# sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Binary classificiation:
# If new to python it is recommended to google all sklearn functions used in this script
# and to try running them with different arguments.
#
# This script will
# - Download a dataset
# - Split dataset into train/test
# - Train and evaluate with RandomForest and Logistic Regression


###
# DOWNLOAD DATASET & PREPROCESSING
###

"""
Our dataset is downloadable as a function in sklearn.

If you have custom data in .csv, .xlsm, or a similar format: 
google pandas read_csv or your .ext as pandas supports most common extensions.
"""

# load dataset from function
X, y = load_breast_cancer(return_X_y=True)

# check size of data and target
print(f"feature data: {X.shape}, target data:{y.shape}")

# split into training and testing data so we can evaluate if our trained model generalizes well to unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###
# CLASSIFICATION
###

"""
Random Forest:
RF uses the best average of decision trees to classify a sample.
A decision tree classifies into 'True' or 'False' by deciding a threshold for each feature variable, starting 
with the most important feature, and sending a sample down the tree of threshold rules to make a decision.
RF does not require normalization.
"""

# init a classifier object
clf = RandomForestClassifier(max_depth=2, random_state=0)

# train object on training data
clf.fit(X_train, y_train)

# get predictions y_hat on testing data
y_test_hat = clf.predict(X_test)

# check accuracy between true and predicted targets
acc = accuracy_score(y_test, y_test_hat)

# print accuracy and format to 3 floats
print(f"Test Acc: [{acc:.3}]")


"""
Logistic regression:
"""
