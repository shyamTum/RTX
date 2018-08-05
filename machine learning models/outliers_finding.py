import pandas as pd 
import numpy as np 
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from scipy.interpolate import *
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from scipy import stats
import re


def find_median(list):
	return np.median(list)

def median_abs_dev(list):
	median = find_median(list)
	return find_median(np.absolute(list - median))
	# return pd.DataFrame.mad(list)

df = pd.read_csv("../examples/crowdnav-sequential/results.csv",header=None)
df = pd.DataFrame(df)
df_x = df.iloc[:,0:2]
df_y = df.iloc[:,3:]
df_x = np.array(df_x)
df_y = np.array(df_y)

print('median',statistics.median(df_y))
print('median_abs_dev()',median_abs_dev(df_y))

threshold1 = find_median(df_y) + median_abs_dev(df_y)*2.5
threshold2 = find_median(df_y) - median_abs_dev(df_y)*2.5
temp_list = []
for x in df_y:
    if x>threshold1 or x<threshold2:
        temp_list.append(x)
print("The outliers:  ",temp_list)
print('########             Outliers points are         ##############')
for x in temp_list:
    print("Outliers rows",df.loc[df[3]==x[0]])
    print('----------------------------------')

def checkOutLiers():

    df = pd.read_csv("../examples/crowdnav-sequential/results.csv",header=None)
    df = pd.DataFrame(df)
    df_x = df.iloc[:,0:2]
    df_y = df.iloc[:,3:]
    df_x = np.array(df_x)
    df_y = np.array(df_y)

    print('median',statistics.median(df_y))
    print('median_abs_dev()',median_abs_dev(df_y))

    threshold1 = find_median(df_y) + median_abs_dev(df_y)*2.5
    threshold2 = find_median(df_y) - median_abs_dev(df_y)*2.5
    temp_list = []
    for x in df_y:
    	if x>threshold1 or x<threshold2:
    		temp_list.append(x)
    print("The outliers:  ",temp_list)
    print('########             Outliers points are         ##############')
    for x in temp_list:
    	print("Outliers rows",df.loc[df[3]==x[0]])
    	print('----------------------------------')

# checkOutLiers()





# 	print('median_abs_dev()',median_abs_dev(df_y))
# 	# print('median_abs_dev pandas',pd.DataFrame.mad(array))
# 	threshold1 = find_median(df_y) + median_abs_dev(df_y)*2
# 	threshold2 = find_median(df_y) - median_abs_dev(df_y)*2
# 	# temp_list = [i for i in array if threshold1<i<threshold2]
# 	temp_list = []
# 	for x in array:
# 		if x>threshold1 or x<threshold2:
# 			temp_list.append(x)
# 	print("The outliers:  ",temp_list)
# 	print('########             Outliers points are         ##############')
# 	for x in temp_list:
# 		print("Outliers rows",df.loc[df[3]==x[0]])
# 		print('----------------------------------')
# 	# print("rows ",df.loc[df[3]==temp_list[0][0]])


# checkOutLiers()

