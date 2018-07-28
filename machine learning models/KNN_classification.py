import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import *
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.cross_validation import KFold
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

knn_clf1 = KNeighborsClassifier(n_neighbors=11)

###############For Other classification#################

# newData = datasets.load_wine()
# print("newData.data", newData)
# X = newData.data
# Y = newData.target
# print("X ",X)
# print("Y ",Y)

# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.5,random_state=4)
#######################################################


knn_clf1.fit(x_train, y_train)

# y_1 = regr_1.predict(x_test)
# y_2 = regr_2.predict(x_test)
k_1 = knn_clf1.predict(x_test)

# print('y_1 predicted  ',y_1)
# print('y_2 predicted  ',y_2)
print('k_1 predicted ',k_1)


print('y_test  ',y_test)
print('x_test  ',x_test)

# print("y_1 mean square error !!!!!!!!! ", mean_squared_error(y_test,y_1))
# print("y_2 mean square error !!!!!!!!! ", mean_squared_error(y_test,y_2))
print("k_1 mean square error !!!!!!!!! ", mean_squared_error(y_test,k_1))


###############Cross Validation - Learning Curve###################

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     return plt

# size = 100
# title = "Learning Curves (Knn Classifiers)"
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# # cv = KFold(size, shuffle=True)
# plot_learning_curve(knn_clf1, title, df_x, df_y, cv=cv)
# plt.show()

###########################################################################

x_test=np.linspace(0,0.3,100)
# y_2=np.linspace(0,5,100)
y_test=np.linspace(0,5,100)
# y_1=np.linspace(0,5,100)
k_1=np.linspace(0,5,100)
# plt.scatter(x_test, y_test, c="darkorange")
# plt.plot(x_test, y_1, color="cornflowerblue")

# plt.plot(x_test, y_2, color="yellowgreen")
# plt.plot(x_test, k_1, color="yellowgreen")
# plt.plot(df_x,df_y,'o')
# plt.xlabel("input")
# plt.ylabel("feedback")
# plt.title("Decision Tree Classification")
# plt.legend()
# plt.show()