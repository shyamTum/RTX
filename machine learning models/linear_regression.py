import pandas as pd 
import numpy as np 
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import axes3d
from sklearn import datasets
# import Gnuplot

 
df = pd.read_csv("../examples/crowdnav-sequential/results.csv",header=None)
df = pd.DataFrame(df)

df_x = df.iloc[:,0:1]
df_y = df.iloc[:,3:]


print("df ",df)
print("df_x ",df_x)
print("df_y ",df_y)

df_x_route = df_x.iloc[:,:1]
df_x_explor = df_x.iloc[:,1:2]


reg = linear_model.LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.4,random_state=4)

reg.fit(x_train,y_train)
a=reg.predict(x_test)
print("\n")
print('input values ',np.array(x_test).flatten())
print("\n")
print("predicted value ",a.flatten())
print("\n")
print("target value ",np.array(y_test).flatten())
print("\n")
# print('a flatten',type(a.flatten()))
# print('extraction!!!!!!!!!!!!',[x1-x2 for x1,x2 in zip(a.flatten(), np.array(y_test))])
# differenceVal = [x1-x2 for x1,x2 in zip(a.flatten(), np.array(y_test))]
# differenceVal = a-np.array(y_test)
# differenceVal = differenceVal.flatten()
# 
# print('difference ',differenceVal)

# x_test_route_randome = x_test.iloc[:,0:1]
# x_test_exploration_rate = x_test.iloc[:,1:2]
# y_test=np.array(y_test)

# print("x_test_route_randome ",x_test_route_randome)
# print("x_test_exploration_rate ",x_test_exploration_rate)

############## 3D plot -wireFrame#################

# x_test_route_randome = np.linspace(0,0.3,100)
# x_test_exploration_rate = np.linspace(0,0.3,100)
# # a=np.arange(-3,3,100)
# # a = np.linspace(-3,3,100)

# fig = plt.figure(figsize=(30,30))
# ax = fig.add_subplot(121, projection='3d')
# ax_ytest = fig.add_subplot(122, projection='3d')

# x_test_route_randome, x_test_exploration_rate, a = axes3d.get_test_data(0.15)
# x_test_route_randome, x_test_exploration_rate, y_test = axes3d.get_test_data(0.15)

# ax.plot_wireframe(x_test_route_randome,x_test_exploration_rate,a, rstride=2, cstride=2)
# ax_ytest.plot_wireframe(x_test_route_randome,x_test_exploration_rate,y_test, rstride=2, cstride=2)
# ax.set_xlabel('route_random_sigma')
# ax.set_ylabel('exploration_rate')
# ax.set_zlabel('predicted value')
# ax_ytest.set_xlabel('route_random_sigma')
# ax_ytest.set_ylabel('exploration_rate')
# ax_ytest.set_zlabel('Y_TEST')
# # ax.set_zlim3d(1.0, 2.0)
# # ax.set_ylim3d(0,0.3)
# # ax.set_xlim3d(0,0.3)

# plt.savefig('3d wireframe')
# plt.show()
#####################################################

##############3D plot - scatter######################

# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(121, projection='3d')
# ax_ytest = fig.add_subplot(122, projection='3d')

# x_test_route_randome, x_test_exploration_rate, a = axes3d.get_test_data(0.05)
# x_test_route_randome, x_test_exploration_rate, y_test = axes3d.get_test_data(0.05)

# ax.scatter(x_test_route_randome,x_test_exploration_rate,a)
# ax_ytest.scatter(x_test_route_randome,x_test_exploration_rate,y_test)
# ax.set_xlabel('route_random_sigma')
# ax.set_ylabel('exploration_rate')
# ax.set_zlabel('predicted value')
# ax_ytest.set_xlabel('route_random_sigma')
# ax_ytest.set_ylabel('exploration_rate')
# ax_ytest.set_zlabel('Y_TEST')
# ax.set_zlim3d(1.0, 2.0)
# ax.set_ylim3d(0,0.3)
# ax.set_xlim3d(0,0.3)

# ax_ytest.set_zlim3d(1.0, 2.0)
# ax_ytest.set_ylim3d(0,0.3)
# ax_ytest.set_xlim3d(0,0.3)

# plt.savefig('scatter graph')

# plt.show()

######################################################


#############3D plot bar################################

# fig = plt.figure(figsize=(20,20))
# ax = fig.add_subplot(121, projection='3d')
# ax_ytest = fig.add_subplot(122, projection='3d')

# # x_test_route_randome, x_test_exploration_rate, a = axes3d.get_test_data(0.05)
# # x_test_route_randome, x_test_exploration_rate, y_test = axes3d.get_test_data(0.05)

# # ax.scatter(x_test_route_randome,x_test_exploration_rate,a)
# # ax_ytest.scatter(x_test_route_randome,x_test_exploration_rate,y_test)

# dx=np.ones(len(x_test_route_randome))
# dy=np.ones(len(x_test_exploration_rate))
# dz=np.ones(len(a))
# ax.bar(x_test_route_randome,x_test_exploration_rate,a,zdir='y',color='green')
# ax_ytest.bar(x_test_route_randome,x_test_exploration_rate,y_test,zdir='y',color='green')

# ax.set_xlabel('route_random_sigma')
# ax.set_ylabel('exploration_rate')
# ax.set_zlabel('predicted value')
# ax_ytest.set_xlabel('route_random_sigma')
# ax_ytest.set_ylabel('exploration_rate')
# ax_ytest.set_zlabel('Y_TEST')
# ax.set_zlim3d(1.0, 2.0)
# ax.set_ylim3d(0,0.3)
# ax.set_xlim3d(0,0.3)

# ax_ytest.set_zlim3d(1.0, 2.0)
# ax_ytest.set_ylim3d(0,0.3)
# ax_ytest.set_xlim3d(0,0.3)

# print("y_train ",y_train)
# print("x_train ",x_train)

# print("coeficients ",reg.coef_)
# print("intercept ",reg.intercept_)
# print("Mean squared error ",mean_squared_error(y_test,a))

# plt.show()

for x, y in zip(x_test,y_test):
   plt.scatter(x_test,y_test, color='black')
for x, y in zip(x_train,y_train):
   plt.scatter(x_train,y_train, color='blue')

###############gnu plot###################

# g=Gnuplot.Gnuplot()
# d1 = Gnuplot.Data(x_test,a,with_='lp', title='d1')
# d2 = Gnuplot.Data(x_test,y_test,with_='lp', title='d2')
# g('set grid')
# g('set key left ')
# g.plot(d1,d2)

##########################################


##############2D plot#################

# print("y_train ",y_train)
# print("x_train ",x_train)

print("coeficients ",reg.coef_)
print("\n")
print("intercept ",reg.intercept_)
print("\n")
print("Mean squared error ",mean_squared_error(y_test,a))
print("\n")


x_test=np.linspace(0,0.3,10)
a=np.linspace(1.5,1.8,10)
y_test=np.linspace(1.5,1.8,100)
# differenceVal=np.linspace(0,1.8,10)


#############plot for x_test vs a and y_test ###############
plt.plot(x_test,a,'g--')
# plt.plot(x_train,a)
# plt.plot(x_test,y_test,'r-')

############################################################

############plot for differences between a and y_test#############
# plt.plot(x_test,differenceVal,'o-')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_test, y_test, a)

plt.xlabel("input")
plt.ylabel("Avg_overhead")

plt.show()






