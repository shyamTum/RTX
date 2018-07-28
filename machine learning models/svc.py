import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn import svm


df=pd.read_csv("examples/crowdnav-sequential/results_multivariate.csv",header=None)
df = pd.DataFrame(df)

# boston = datasets.load_iris()
# print("boston.data", boston.data)
# dataframe = pandas.read_csv(url, names=names)
# array = dataframe.values
X = df.iloc[:,0:2]
Y = df.iloc[:,2:3].values.ravel()
# X = boston.data
# Y = boston.target
print("X ",X)
print("Y ",Y)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=4)

###############For Other classification#################

# newData = datasets.load_wine()
# print("newData", newData.data)
# X = newData.data
# Y = newData.target
# print("X ",X)
# print("Y ",Y)

# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=4)

#############################################################

model = svm.SVC(kernel='rbf', C=4,gamma='auto').fit(x_train, y_train)
# model = svm.SVC()

a= model.predict(x_test)

print("predicted a",a)

print("y_test ",y_test)

print("mean_squared_error ",mean_squared_error(y_test,a))