import numpy as np
from scipy import signal
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy import optimize
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("examples/crowdnav-sequential/results.csv",header=None)
df = pd.DataFrame(df)

X = df.iloc[:,0:2]
Y = df.iloc[:,2:3]

df_x = np.array(X)
df_y = np.array(Y)

reg = linear_model.LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.4,random_state=4)


def find_peak(model,X1,Y1,key):
	print("############ using optimize function ################")
	print("\n")
	reg1=model
	reg1.fit(x_train,y_train)
	x0 = [df_x]
	def f(x):
		return reg1.predict([x])
	optimizedVal = optimize.minimize(f,(0,0),method='TNC',bounds=((0.0,0.3),(0.0,0.3)))
	print('\n')
	print("optimized points == ",optimizedVal.x)


