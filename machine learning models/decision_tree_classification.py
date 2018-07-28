import pandas as pd 
import numpy as np 

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import *
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.cross_validation import KFold
from sklearn import tree
import pydotplus
from sklearn import datasets
from sklearn.metrics import accuracy_score

df = pd.read_csv("../examples/crowdnav-sequential/results.csv",header=None)
df = pd.DataFrame(df)

df_x = df.iloc[:,0:2]
df_y = df.iloc[:,2:3]

df_x = np.array(df_x)
df_y = np.array(df_y)
# df_x = df_x.flatten()
# df_y = df_y.flatten()

print("df ",df)
print("input variables ",df_x)
print("target values ",df_y)

x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.5,random_state=4)

regr_1 = DecisionTreeClassifier(max_depth=2)
regr_2 = DecisionTreeClassifier(max_depth=4)

regr_1.fit(x_train, y_train)
regr_2.fit(x_train, y_train)

############### Decision tree structure output####################

dot_data1=tree.export_graphviz(regr_1,out_file='tree1.dot')
dot_data2=tree.export_graphviz(regr_2,out_file='tree2.dot')


##############################################


y_1 = regr_1.predict(x_test)
y_2 = regr_2.predict(x_test)
# y_3 = regr_3.predict(x_test)

print("\n")
print('predicted values for max_depth = 2  ',y_1)
print("\n")
print('predicted values for max_depth = 4  ',y_2)
print("\n")

print('target values  ',y_test.flatten())
print("\n")

print("Accuracy score for max_depth = 2 ==>", accuracy_score(y_test,y_1))
print("Accuracy score for max_depth = 4 ==>", accuracy_score(y_test,y_2))


##############Cross Validation - Learning Curve###################

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1, 50)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

title = "Learning Curves (Decision Tree Classifiers)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
plot_learning_curve(regr_1, title, df_x, df_y, cv=cv)
plt.show()

##########################################################################

