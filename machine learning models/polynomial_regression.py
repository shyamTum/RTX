import pandas as pd 
import numpy as np 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import *
from sklearn.metrics import mean_squared_error
from sklearn import datasets

df = pd.read_csv("examples/crowdnav-sequential/results_multivariate.csv",header=None)
df = pd.DataFrame(df)
print("df ",df)

# df_x = df.iloc[:,0:1]
# df_y = df.iloc[:,2:]

df_x = df.iloc[:,0:1]
df_y = df.iloc[:,3:]

df_x = np.array(df_x)
df_y = np.array(df_y)
# df_x = df_x.flatten()
# df_y = df_y.flatten()
# df_x = df_x.reshape(4,2)

# print("df ",df)
df_x=df_x[:,0]
df_y=df_y[:,0]

print("df_x ",df_x)
print("df_y ",df_y)

# df_x = np.array([0,1,2,3,4,5])
# df_y = np.array([0,0.8,0.9,0.1,-0.8,-1])

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.4,random_state=4)
##############Other regression dataset##########
# newData = datasets.load_diabetes()
# print("newData.data", newData.data)
# X = newData.data
# Y = newData.target
# X = np.array(X)
# Y = np.array(Y)
# X = X.flatten()
# # Y = Y.flatten()
# # X = X.reshape(X[1:])
# # Y=Y[442:]
# # Y.reshape((442,1))
# print("X ",X)
# print("Y ",Y)

# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=4)
########################################################################


# for x, y in zip(x_test,y_test):
#    plt.scatter(x_test,y_test, color='black')
# for x, y in zip(x_train,y_train):
#    plt.scatter(x_train,y_train, color='blue')
# print("x_train!!!!!!! ",x_train)
# print("y_train!!!!!!! ",y_train)
# print("x_test!!!!!!! ",x_test)
print("\n")
print("y_test!!!!!!! ",y_test)
print("\n")

# poly1=np.polyfit(x_train,y_train,1)
poly2=np.polyfit(x_train,y_train,2)
poly3=np.polyfit(x_train,y_train,3)
poly4=np.polyfit(x_train,y_train,4)
poly5=np.polyfit(x_train,y_train,5)
poly6=np.polyfit(x_train,y_train,6)
# poly7=np.polyfit(x_train,y_train,7)


# print('poly1 coefficients ',poly1)
print('poly2 coefficients ',poly2)
print("\n")
print('poly3 coefficients ',poly3)
print("\n")

# a2=poly2.predict(x_test)
# a3=poly3.predict(x_test)

# plt.plot(x_test,y_test)
plt.plot(df_x,df_y,'o')

# print("Mean squared error for linear regr!!!!!!",mean_squared_error(y_test, np.polyval(poly1,x_test)))
print("Mean squared error for degree 2 polynomial regr!!!!!!",mean_squared_error(y_test, np.polyval(poly2,x_test)))
print("\n")
print("Mean squared error for degree 3 polynomial regr!!!!!!",mean_squared_error(y_test, np.polyval(poly3,x_test)))
print("\n")
# print("Mean squared error for degree 4 polynomial regr!!!!!!",mean_squared_error(y_test, np.polyval(poly4,x_test)))
# print("Mean squared error for degree 5 polynomial regr!!!!!!",mean_squared_error(y_test, np.polyval(poly5,x_test)))
# print("Mean squared error for degree 6 polynomial regr!!!!!!",mean_squared_error(y_test, np.polyval(poly6,x_test)))
# print("linear predicted values!!!!!!",np.polyval(poly1,x_test))
print("y_test!!!!!!! ",y_test)
print("\n")
print("degree 2 polynomial predicted values!!!!!!",np.polyval(poly2,x_test))
print("\n")
print("degree 3 polynomial predicted values!!!!!!",np.polyval(poly3,x_test))
print("\n")
# print("degree 4 polynomial predicted values!!!!!!",np.polyval(poly4,x_test))
# print("degree 5 polynomial predicted values!!!!!!",np.polyval(poly5,x_test))
# print("degree 6 polynomial predicted values!!!!!!",np.polyval(poly6,x_test))

# print("x_test!!!!!!! ",x_test)
# print("y_test!!!!!!! ",y_test)


x_test=np.linspace(0,0.3,100)

# plt.plot(x_test,np.polyval(poly1,x_test),'b-',label='line for degree 1')
plt.plot(x_test,np.polyval(poly2,x_test),'r-',label='line for degree 2')
plt.plot(x_test,np.polyval(poly3,x_test),'black',label='line for degree 3')
plt.plot(x_test,np.polyval(poly4,x_test),'green',label='line for degree 4')
# plt.plot(x_test,np.polyval(poly5,x_test),'orange')
# plt.plot(x_test,np.polyval(poly6,x_test),'pink')
plt.legend()

plt.xlabel("route_random_sigma")
plt.ylabel("Avg_overhead")

plt.show()
