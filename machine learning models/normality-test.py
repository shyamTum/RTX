import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import *
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from scipy import stats



df = pd.read_csv("examples/crowdnav-sequential/results_multivariate.csv",header=None)
df = pd.DataFrame(df)
# print("df ",df)

df_x = df.iloc[:,0:2]
df_y = df.iloc[:,2:3]

df_x = np.array(df_x)
df_y = np.array(df_y)

print(stats.shapiro(df_x))

stats.probplot(df_x, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()

# measurements = np.random.normal(df_x)   
# stats.probplot(measurements, dist="norm", plot=plt)
# plt.show()