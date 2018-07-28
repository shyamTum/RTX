import pandas as pd 
import numpy as np 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import *
from sklearn import preprocessing as pp
from sklearn.metrics import mean_squared_error
from sklearn import datasets

df = pd.read_csv("examples/crowdnav-sequential/results_multivariate.csv",header=None)
df = pd.DataFrame(df)

df_x = df.iloc[:,0:2]
df_y = df.iloc[:,3:]

df_x = np.array(df_x)
df_y = np.array(df_y)
# df_x = df_x.flatten()
# df_y = df_y.flatten()

print("df ",df)
print("df_x ",df_x)
print("df_y ",df_y)

# df_x = np.array([0,1,2,3,4,5])
# df_y = np.array([0,0.8,0.9,0.1,-0.8,-1])

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.5,random_state=4)

##############Other regression dataset##########
# newData = datasets.load_diabetes()
# print("newData.data", newData.data)
# X = newData.data
# Y = newData.target

# print("X ",X)
# print("Y ",Y)

# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=4)
########################################################################

# for x, y in zip(x_test,y_test):
#    plt.scatter(x_test,y_test, color='black')
# for x, y in zip(x_train,y_train):
#    plt.scatter(x_train,y_train, color='blue')


regression1 = pp.PolynomialFeatures(degree=1)
x_train_=regression1.fit_transform(x_train)
x_test_ = regression1.fit_transform(x_test)
# a=regression.predict(x_test)

lr1 = linear_model.LinearRegression()
lr1.fit(x_train_,y_train)
a1=lr1.predict(x_test_)


regression2 = pp.PolynomialFeatures(degree=2)
x_train_=regression2.fit_transform(x_train)
x_test_ = regression2.fit_transform(x_test)
# a=regression.predict(x_test)

lr2 = linear_model.LinearRegression()
lr2.fit(x_train_,y_train)
a2=lr2.predict(x_test_)


regression4 = pp.PolynomialFeatures(degree=4)
x_train_=regression4.fit_transform(x_train)
x_test_ = regression4.fit_transform(x_test)
# a=regression.predict(x_test)

lr4 = linear_model.LinearRegression()
lr4.fit(x_train_,y_train)
a4=lr4.predict(x_test_)


regression5 = pp.PolynomialFeatures(degree=5)
x_train_=regression5.fit_transform(x_train)
x_test_ = regression5.fit_transform(x_test)
# a=regression.predict(x_test)

lr5 = linear_model.LinearRegression()
lr5.fit(x_train_,y_train)
a5=lr5.predict(x_test_)

print("\n")
print("y_test ",y_test.flatten())
print("\n")
# print("x_test ",x_test)
# print("coefficients for deg 1 ",regression1.get_feature_names())
# print("coefficients for deg 2 ",regression2.get_feature_names())
# print("coefficients for deg 4 ",regression4.get_feature_names())
# print("coefficients for deg 5 ",regression5.get_feature_names())
print("degree1 prediction ",a1.flatten())
print("\n")
print("degree2 prediction ",a2.flatten())
print("\n")
print("degree4 prediction ",a4.flatten())
print("\n")
print("degree5 prediction ",a5.flatten())
print("\n")

print("mean_squared_error for degree 1 ",mean_squared_error(y_test,a1))
print("mean_squared_error for degree 2 ",mean_squared_error(y_test,a2))
print("mean_squared_error for degree 4 ",mean_squared_error(y_test,a4))
print("mean_squared_error for degree 5 ",mean_squared_error(y_test,a5))

#############################################
# print("final coefficients for degree 1 ",pd.DataFrame(regression1.transform(x_test), columns=regression1.get_feature_names()))
# print("final coefficients for degree 2 ",pd.DataFrame(regression2.transform(x_test), columns=regression2.get_feature_names()))
# print("final coefficients for degree 4 ",pd.DataFrame(regression4.transform(x_test), columns=regression4.get_feature_names()))
# print("final coefficients for degree 5 ",pd.DataFrame(regression5.transform(x_test), columns=regression5.get_feature_names()))
# print("poly powers deg 2",regression2.powers_)
# print("poly powers deg 5",regression5.powers_)



x_test=np.linspace(0,4.5,10)
a1=np.linspace(0,4.5,10)
a2=np.linspace(0,4.5,10)
a4=np.linspace(0,4.5,10)

plt.plot(x_test,a1,'brown')
plt.plot(x_test,a2,'ro-')
plt.plot(x_test,a4,'blue')

plt.xlabel("route_random_sigma + explor. rate")
plt.ylabel("Avg_overhead")

plt.show()

