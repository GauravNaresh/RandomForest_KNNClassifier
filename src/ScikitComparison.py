
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import BaggingClassifier as BG
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


cancer = load_breast_cancer()

#Scaling the features between 0 and 1.
scaler=StandardScaler()
scaler.fit(cancer.data)
X_scaled=scaler.transform(cancer.data)

print("after scaling minimum", X_scaled.min(axis=0))

pca = PCA(n_components=6)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)
#X contains X_pca
X=X_pca
Y=cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

score = "accuracy"

print("******************************RANDOM FORESTS****************************")

RFC_tuned_parameter = [{'n_estimators': [5,10,15,20,50], 'max_depth': [5,10,15,20], 'max_features':[1,2,3],
                        'criterion': ['gini','entropy'], 'min_samples_split': [2,3,4,5,6],
                        'min_samples_leaf': [1,2,3,4,5,6,7]}]


clf = GridSearchCV(RFC(), RFC_tuned_parameter, cv=6, scoring ='%s' %score)

clf.fit(X_train, y_train)

print()
print("Grid scores on training data:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()

print("The classification report for SKlearn's Random Forests   are as follows")
print()
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()


print("*******************************KNN******************************************")

KNN_tuned_parameter = [{'n_neighbors': [3,5,7,10], 'algorithm': ['auto','ball_tree','kd_tree','brute'],
                        'p': [1,2,3], 'weights': ['uniform', 'distance']}]

clf = GridSearchCV(KNN(), KNN_tuned_parameter, cv=6, scoring='%s' % score)
clf.fit(X_train, y_train)

print()
print("Grid scores on training data:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()

print("The classification report for SKlearn's KNN   are as follows")
print()
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()