# -----------------------------------------------------------
# classical methods for binary classification in sklearn
#
# Torjus Nilsen, Kongsberg, Norway
# email tornil1996@hotmail.com
# -----------------------------------------------------------

# preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# inference
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
# DOWNLOAD DATASET
###

def get_dataset():
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

	return X_train, X_test, y_train, y_test

###
# CLASSIFICATION
###

def run_randomforest(X_train, X_test, y_train, y_test):
	"""
	RF uses the best average of decision trees to classify a sample.
	A decision tree classifies into 'True' or 'False' by deciding a threshold for each feature variable, starting 
	with the most important feature, and sending a sample down the tree of threshold rules to make a decision.
	RF does not require normalization.
	"""

	# init a classifier object
	clf_rf = RandomForestClassifier(max_depth=2, random_state=0)

	# train object on training data
	clf_rf.fit(X_train, y_train)

	# get predictions y_hat on testing data
	y_test_hat = clf_rf.predict(X_test)

	# check accuracy between true and predicted targets
	acc_rf = accuracy_score(y_test, y_test_hat)

	return acc_rf

def run_logreg(X_train, X_test, y_train, y_test):
	"""
	Default arguments of logreg does not converge for this dataset.

	A good exercise could be to first try LogisticRegression() 
	without ags, then trying to fix it from sklearns list of possible solutions.

	Things to help convergence could be:
		1. Normalizing input
		2. Trying a different solver
		3. Increasing maximum number of iterations
	"""

	# init a classifier object
	clf_logreg = LogisticRegression()

	# 1. normalize input
	X_train = normalize(X_train)

	# 2. different solver
	#clf_logreg = LogisticRegression(solver="newton-cg")

	# 3. sgd solver that requires more iters
	#clf_logreg = LogisticRegression(solver="sag", max_iter=1000)

	# train object on training data
	clf_logreg.fit(X_train, y_train)

	# get predictions y_hat on testing data
	y_test_hat = clf_logreg.predict(X_test)

	# check accuracy between true and predicted targets
	acc_logreg = accuracy_score(y_test, y_test_hat)

	return acc_logreg

###
# MAIN
###

def main():

	X_train, X_test, y_train, y_test = get_dataset()
	acc_rf = run_randomforest(X_train, X_test, y_train, y_test)
	acc_logreg = run_logreg(X_train, X_test, y_train, y_test)

	# print accuracy and format to 3 floats
	print(f"RandomForest Test Acc: [{acc_rf:.3}]")
	print(f"LogReg Test Acc: [{acc_logreg:.3}]")


# boilerplate (common but not needed) pythonic way of only running script when meant to
if __name__ == "__main__":
    main()