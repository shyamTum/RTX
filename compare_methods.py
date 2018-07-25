import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing as pp
from sklearn import datasets
from numpy.polynomial.polynomial import polyval
import numpy.polynomial.polynomial as poly
from sklearn.cross_validation import train_test_split
from rtxlib import info
from colorama import Fore
# from outliers_finding import checkOutLiers

df=pd.read_csv("examples/crowdnav-sequential/results.csv",header=None)
df = pd.DataFrame(df)

########Check for continuous or discrete data ###################
def checkContinuous(array):
	for y in array:
		if(isinstance(y,float)):
			print('\n')
			print ("Regression Methods!!!!!!!!!!!!!!!!!")
			return True
		else:
			print('\n')
			print("Classification Methods!!!!!!!!!!!")
			return False

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
    	makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


def classifier_compare_methods():

	############CrowdNav dataset#########################
	X = df.iloc[:,0:2]
	Y = df.iloc[:,2:3].values.ravel()
	
	seed = 7

	################################# prepare models #########################
	models = []
	if not checkContinuous(Y):
		models.append(('LR', LogisticRegression()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('Dec. Tree', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC(kernel='linear', C=2,gamma='auto')))
		scoring = 'accuracy'
		# scoring = 'average_precision'

	else:
		return False

	#######################evaluate each model in turn#####################
	results = []
	names = []
	axisLimit = []
	selectMethod = {}
	selectMethodName = 0
	finalSelectMethod = {}
	messages = []
	info('##################################################', Fore.CYAN)
	info('    Analysis for Avg_Feedback (Classification)     #')
	info('----------------------------------------------------', Fore.CYAN)
	print('Method Name | CV_result_mean')
	print('---------------------------------------------------')
	for name, model in models:
		kfold = model_selection.KFold(n_splits=3, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

		results.append(cv_results)
		names.append(name)
		selectMethod[name]=cv_results.mean()
		msg = "%s: %f (std_dev=%f)" % (name, cv_results.mean(), cv_results.std())
		msg_new = "%s  | %f " % (name, cv_results.mean())
		
		messages.append(msg_new)
		print(msg_new)

	# #############################boxplot algorithm comparison##################

	################CrowdNav####################
	finalMethodSelectValue = max(selectMethod.values())
	for key in selectMethod:
		if(selectMethod[key] == finalMethodSelectValue):
			finalSelectMethod[key] = selectMethod[key]
	
	print("\n")
	info('--------------------------------------------------\n')
    
	info('##################################################', Fore.CYAN)
	info('#    Chosen Data Driven Method (Classification)     #')
	info('--------------------------------------------------', Fore.CYAN)
	print(finalSelectMethod)
	print("\n")
	info('---------------------------------------------------', Fore.CYAN)
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	output_dir = "./result_graphs"
	mkdir_p(output_dir)	
	plt.savefig('./result_graphs/classifier.png')


def findModelName(finalSelectMethod):
	X = df.iloc[:,0:2]
	Y = df.iloc[:,2:3].values.ravel()
	if list(finalSelectMethod.keys())==['LR']:
		return LogisticRegression(),X,Y
	elif list(finalSelectMethod.keys())==['KNN']:
		return KNeighborsClassifier(),X,Y
	elif list(finalSelectMethod.keys())==['Dec. Tree Classifier']:
		return DecisionTreeClassifier(),X,Y
	elif list(finalSelectMethod.keys())==['NB']:
		return GaussianNB(),X,Y
	elif list(finalSelectMethod.keys()) == ['SVM']:
		return SVC(kernel='linear', C=2,gamma='auto'),X,Y
	elif list(finalSelectMethod.keys())==['Lin. Reg.']:
		return LinearRegression(),X,Y
	elif list(finalSelectMethod.keys())==['Poly2']:
		regression2 = pp.PolynomialFeatures(degree=2)
		X_pol2=regression2.fit_transform(X)
		return LinearRegression(),X_pol2,Y
	elif list(finalSelectMethod.keys())==['Poly3']:
		regression3 = pp.PolynomialFeatures(degree=3)
		X_pol3=regression3.fit_transform(X)
		return LinearRegression(),X_pol3,Y
	elif list(finalSelectMethod.keys())==['Poly4']:
		regression4 = pp.PolynomialFeatures(degree=4)
		X_pol4=regression4.fit_transform(X)
		return LinearRegression(),X_pol4,Y
	elif list(finalSelectMethod.keys())==['Dec. Tree regressor']:
		return DecisionTreeRegressor(),X,Y




def regressor_compare_methods():

	############CrowdNav dataset#########################
	X = df.iloc[:,0:2]
	Y = df.iloc[:,3:].values.ravel()

	########handling polynomial regression##################

	regression2 = pp.PolynomialFeatures(degree=2)
	X_pol2=regression2.fit_transform(X)

	regression3 = pp.PolynomialFeatures(degree=3)
	X_pol3=regression3.fit_transform(X)

	regression4 = pp.PolynomialFeatures(degree=4)
	X_pol4=regression4.fit_transform(X)

	regression5 = pp.PolynomialFeatures(degree=5)
	X_pol5=regression5.fit_transform(X)

	seed = 7

	
	################################# prepare models #########################
	models = []
	if not checkContinuous(Y):
		return False

	else:
	    models.append(('Lin. Reg.',LinearRegression()))
	    models.append(('Poly2',LinearRegression()))
	    models.append(('Poly3',LinearRegression()))
	    models.append(('Poly4',LinearRegression()))
	    models.append(('Dec. Tree regressor', DecisionTreeRegressor()))
	    scoring = 'neg_mean_squared_error'
	    # scoring = 'explained_variance'
	    # scoring = 'r2'

	#######################evaluate each model in turn#####################
	results = []
	names = []
	axisLimit = []
	selectMethod = {}
	selectMethodName = 0
	finalSelectMethod = {}
	messages = []
	print("\n")
	info('##################################################', Fore.CYAN)
	info('       Analysis for Avg_Overhead (Regression)     ', Fore.CYAN)
	info('--------------------------------------------------', Fore.CYAN)
	print('Method Name | CV_result_mean')
	print('-------------------------------------------------')
	for name, model in models:
		kfold = model_selection.KFold(n_splits=3, random_state=seed)
		if name=='Poly2':
			cv_results = model_selection.cross_val_score(model, X_pol2, Y, cv=kfold, scoring=scoring)
		elif name=='Poly3':
			cv_results = model_selection.cross_val_score(model, X_pol3, Y, cv=kfold, scoring=scoring)
		elif name=='Poly4':
			cv_results = model_selection.cross_val_score(model, X_pol4, Y, cv=kfold, scoring=scoring)
		else:
			cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

		results.append(cv_results)
		names.append(name)
		selectMethod[name]=cv_results.mean()
		msg = "%s: %f (std_dev=%f)" % (name, cv_results.mean(), cv_results.std())
		msg_new = "%s  | %f \n" % (name, cv_results.mean())
		print(msg_new)

	# #############################boxplot algorithm comparison##################

	################CrowdNav####################
	finalMethodSelectValue = max(selectMethod.values())
	for key in selectMethod:
		if(selectMethod[key] == finalMethodSelectValue):
			finalSelectMethod[key] = selectMethod[key]
	info('--------------------------------------------------\n')
	info('##################################################', Fore.CYAN)
	info('          Best data driven method (Regression)      ', Fore.CYAN)
	info('--------------------------------------------------', Fore.CYAN)
	print(finalSelectMethod)
	print('\n')
	finalSelectModel,finalDataX,finalDataY = findModelName(finalSelectMethod)
	from find_peak import find_peak
	find_peak(finalSelectModel,finalDataX,finalDataY,list(finalSelectMethod.keys()))
	print('\n')

	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	
	output_dir = "./result_graphs"
	mkdir_p(output_dir)
	plt.savefig('./result_graphs/regressor.png')

	
