import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import *
from sklearn.metrics import mean_squared_error
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

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.5,random_state=4)

###############For Other classification#################

# newData = datasets.load_wine()
# print("newData", newData.data)
# X = newData.data
# Y = newData.target
# print("X ",X)
# print("Y ",Y)

# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=4)

#############################################################

# for x, y in zip(x_test,y_test):
#    plt.scatter(x_test,y_test, color='black')
# for x, y in zip(x_train,y_train):
#    plt.scatter(x_train,y_train, color='blue')

# y_train=y_train.flatten()
print("y_train!!!!!!",y_train)
print("y_train as integer!!!!!!",y_train.astype('int'))
# print("y_train as integer!!!!!!",[1 if d>2.3 else 0 for d in y_train])

logReg= LogisticRegression()
logReg.fit(x_train,y_train)
# logReg.fit(x_train,[1 if d>2.3 else 0 for d in y_train])
a=logReg.predict(x_test)
print("a!!!!!! ",a)
print("y_test!!!!!!",y_test)
print("mean square error !!!!!!!!! ", mean_squared_error(y_test,a))

plt.plot(df_x,df_y,'o')
plt.plot(x_test,a)
plt.show()

# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# # %matplotlib inline

# cancer = load_breast_cancer()
# print(cancer.data)

# x_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)


# print(y_train)
# log_reg = LogisticRegression()
# log_reg.fit(x_train, y_train)

# import pandas as pd 
# import numpy as np 
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
# import matplotlib.pyplot as plt
# from scipy.interpolate import *
# from sklearn.metrics import mean_squared_error
# from sklearn.utils import shuffle

# df = pd.read_csv("examples/crowdnav-sequential/results.csv",header=None)
# df = pd.DataFrame(df)
# # print("df ",df)

# df_x = df.iloc[:,0:1]
# df_y = df.iloc[:,1:2]

# df_x = np.array(df_x)
# df_y = np.array(df_y)
# # df_x = df_x.flatten()
# # df_y = df_y.flatten()

# print("df ",df)
# print("df_x ",df_x)
# print("df_y ",df_y)

# x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.5,random_state=4)

# for x, y in zip(x_test,y_test):
#    plt.scatter(x_test,y_test, color='black')
# for x, y in zip(x_train,y_train):
#    plt.scatter(x_train,y_train, color='blue')

# # y_train=y_train.flatten()
# print("y_train!!!!!!",y_train)
# print("y_train as integer!!!!!!",y_train.astype('int'))

# logReg= LogisticRegression()

# x_train,y_train=shuffle(x_train,y_train)

# logReg.fit(x_train,y_train.astype('int'))
# a=logReg.predict(x_test)
# print("a!!!!!! ",a)
# print("y_test!!!!!!",y_test)
# print("mean square error !!!!!!!!! ", mean_squared_error(y_test,a))

# plt.plot(df_x,df_y,'o')
# plt.plot(x_test,a)
# plt.show()