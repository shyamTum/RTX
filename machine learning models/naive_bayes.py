import pandas as pd 
import numpy as np 

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.cross_validation import KFold
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

df = pd.read_csv("../examples/crowdnav-sequential/results.csv",header=None)
df = pd.DataFrame(df)
# print("df ",df)

df_x = df.iloc[:,0:2]
df_y = df.iloc[:,2:3]

df_x = np.array(df_x)
df_y = np.array(df_y)
# df_x = df_x.flatten()
# df_y = df_y.flatten()

print("df ",df)
print("df_x ",df_x)
print("df_y ",df_y)

# x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.5,random_state=4)

###############For Other classification#################

newData = datasets.load_wine()
print("newData.data", newData.data)
X = newData.data
Y = newData.target
print("X ",X)
print("Y ",Y)

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=4)
#######################################################

gnb = GaussianNB()

a = gnb.fit(x_train, y_train).predict(x_test)

print("prediction a",a)
print("y_test ",y_test)

print("mean_squared_error",mean_squared_error(y_test,a))

