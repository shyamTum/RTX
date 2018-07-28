## Import the packages
import numpy as np
from scipy import stats
import pandas as pd


## Define 2 random distributions
#Sample Size
N = 10
#Gaussian distributed data with mean = 2 and var = 1
# a = np.random.randn(N) + 2
#Gaussian distributed data with with mean = 0 and var = 1
# b = np.random.randn(N)
a =[1,3,7,10,8,9,2,20,22,12]
b =[1,3,7,10,8,9,2,20,22,12]

df = pd.read_csv("examples/crowdnav-sequential/results_ttest.csv",header=None)
df = pd.DataFrame(df)

# df_x = df.iloc[:,0:2]
df_y = df.iloc[25:,3:]
df_x = df.iloc[:25,3:]
# df_y = df.iloc[3:7,3:]
# df_x = df.iloc[:3,3:]

df_x = np.array(df_x)
df_y = np.array(df_y)

# print("df ",df)
print("\n")
print("Group A = ",df_x.flatten())
print("\n")
print("Group B = ",df_y.flatten())

## Calculate the Standard Deviation
#Calculate the variance to get the standard deviation

#For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
var_a = df_x.var(ddof=1)
var_b = df_y.var(ddof=1)

#std deviation
# s = np.sqrt((var_a + var_b)/2)
# s = np.sqrt((a+b)/2)
s = np.sqrt((var_a + var_b)/2)


## Calculate the t-statistics
# t = (a.mean() - b.mean())/(s*np.sqrt(2/N))
t = (np.mean(df_x) - np.mean(df_y))/(s*np.sqrt(2/N))



## Compare with the critical t-value
#Degrees of freedom
dfTemp = 2*N - 2

#p-value after comparison with the t 
p = 1 - stats.t.cdf(t,df=dfTemp)

# print(a)
# print(b)
print("\n")
print("\n")
print("If the p-value is larger than 0.05, we cannot conclude that a significant difference exists. ")
# print("t-statistics = " + str(t))
# print("p-statistics = " + str(2*p))
#Note that we multiply the p value by 2 because its a twp tail t-test
### You can see that after comparing the t statistic with the critical t value (computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis and thus it proves that the mean of the two distributions are different and statistically significant.


## Cross Checking with the internal scipy function
t2, p2 = stats.ttest_ind(df_x,df_y)
# t2, p2 = stats.ttest_ind(a,b)
print('\n')
print("t = " + str(t2))
print('\n')
print("p = " + str(2*p2))

print('\n')
print("mean of df_x ",np.mean(df_x))
print("mean of df_y ",np.mean(df_y))