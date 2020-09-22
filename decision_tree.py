'''
A decision tree trained to predict the titanic survivors
'''

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from preprocessing import feat_eng

# Load Data
train = pd.read_csv('./Data/train.csv')
Test = pd.read_csv('./Data/test.csv')

# Split data into X and y
y = np.array(train['Survived'])
X = train.drop(columns=['Survived'])

# Feature Engineer on X
X = feat_eng(X)
Test = feat_eng(Test)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# Train a decision tree using grid search and cross validation
dt = DecisionTreeClassifier(random_state=13, criterion='gini')
parameters = {'max_depth':(None, 1, 3, 5, 10, 25), 'min_samples_split':(2, 4, 8, 16, 32), 'max_features':['sqrt', 'log2'], 'min_impurity_decrease':[0.0, 0.01, 0.05, 0.1, 0.5, 1]} # Set parameters to be used in gridsearch
clf = GridSearchCV(dt, parameters, cv=3) # perform gridsearch and cross validation
clf.fit(X_train.iloc[:,1:], y_train)

# Check how well this model predicts our training sample
y_insample = clf.predict(X_train.iloc[:,1:])
acc_in = accuracy_score(y_train, y_insample)
recall_in = recall_score(y_train, y_insample)
prec_in = precision_score(y_train, y_insample)
cm_in = confusion_matrix(y_train, y_insample)
tn_in, fp_in, fn_in, tp_in = confusion_matrix(y_train, y_insample).ravel()


# Get predictions for test set
y_pred = clf.predict(X_test.iloc[:, 1:])
acc = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Make predictions for submission
y_final = clf.predict(Test.iloc[:,1:])
sub = Test[['PassengerId']].astype(int) # Kaggle expects integer
sub['Survived'] = y_final
sub.to_csv('./Predictions/dt_submission.csv', index=False)
