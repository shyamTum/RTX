import numpy as np
from scipy import stats
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing as pp
from sklearn import datasets
from numpy.polynomial.polynomial import polyval
import numpy.polynomial.polynomial as poly
from sklearn.cross_validation import train_test_split
# from rtxlib import info


df=pd.read_csv("../examples/crowdnav-sequential/results.csv",header=None)
df = pd.DataFrame(df)

X = df.iloc[:,0:2]
Y = df.iloc[:,2:3].values.ravel()
print("X ",X)
print("Y ",Y)
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
scoring = 'accuracy'
cv_results_logistic = model_selection.cross_val_score(LogisticRegression(), X, Y, cv=kfold, scoring=scoring)
cv_results_SVC = model_selection.cross_val_score(SVC(), X, Y, cv=kfold, scoring=scoring)

print('cv_results_logistic',cv_results_logistic)
print('cv_results_SVC',cv_results_SVC)

t,p = stats.ttest_ind(cv_results_logistic,cv_results_SVC, axis=0, equal_var=True)

print("p value = ",p)
print("t value = ",t)

if((p/2)<= 0.05):
	print ("Optimized method is SVC")
else:
	print ("Optimized method is LogisticRegression")
	