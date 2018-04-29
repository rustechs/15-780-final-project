#!/usr/bin/env python3

import sys
import numpy as np
import scipy as sp 
import sklearn as skl
from sklearn import linear_model
import time
import itertools as it
import matplotlib.pyplot as plt

debug=1
x=0 #Inputs
y=1 #Labels

def dbg(msg):
	if debug==1:
		print(msg)
	return

class SI():
	regressionData=None
	predicted_model=None
	model_prediction_time_secs=0
	degree_of_polynomial=1
	types_of_nonlinearity=[-1, -1, -1] #[x1.x2, exp(x), log(x)]
	no_original_ip=-1
	regularization_strength=0

	def __init__(self, matlab_api=None, input_data_file=None, degree=-1, regularization=0, xdata=None, ydata=None):
		self.regressionData=[[]]
		self.regressionData+=[[]]
		if input_data_file != None:
			self.readFile(input_data_file)
		else:
			if xdata!=None and ydata!=None:
				for j in range(len(xdata)):
					self.regressionData[x] += [[float(i) for i in xdata[j]]]
					self.regressionData[y] += [[float(i) for i in ydata[j]]]
		if self.regressionData==None:
			sys.exit('No input data found')
		self.degree_of_polynomial=degree
		self.regularization_strength=regularization
		
		plt.figure(1)
		plt.subplot(211)
		plt.plot(self.regressionData[x])
		plt.subplot(212)
		plt.plot(self.regressionData[y])
		plt.show()


	def readFile(self, fname):
		fhandle = open(fname, 'r')
		temp = fhandle.readlines()[1:]
		dataSet=[]
		for t in temp:
			dataSet += [t.strip('\n').split()[0:5]]
		fhandle.close()

	def solver(self, name): #polynomial, curve, ml_regression, robust
		start = time.time()
		if name=='polynomial':
			self.predicted_model = np.polyfit(self.regressionData[x], self.regressionData[y], self.degree_of_polynomial)
		else:
			if name=='curve':
				self.predicted_model = sp.optimize.curve_fit(self.model_function_to_fit, self.regressionData[x], self.regressionData[y])
			else:
				if name=='ml_regression':
					regression = skl.linear_model.LinearRegression()
					regression.fit(self.regressionData[x], self.regressionData[y])
					self.predicted_model = [regression.coef_, regression.intercept_]
				else:
					if name=='robust':
						self.predicted_model = skl.linear_model.RANSACRegressor.fit(self.regressionData[x], self.regressionData[y]).get_params()
					else:
						if name=='l2':
							self.predicted_model = skl.linear_model.Ridge(alpha=self.regularization_strength).fit(self.regressionData[x], self.regressionData[y]).get_params()
						else:
							if name=='l1':
								self.predicted_model = skl.linear_model.Lasso(alpha=self.regularization_strength).fit(self.regressionData[x], self.regressionData[y]).get_params()
							else:
								sys.exit('Invalid solver option')
		self.model_prediction_time_secs = time.time()-start
		dbg('Model predicted as {} in time {} seconds'.format(self.predicted_model, self.model_prediction_time_secs))
		return(self.model_prediction_time_secs, self.predicted_model)

	def getOutput(self):
		#write to file or output to api
		fmodel = open(Model.txt, 'w+')
		fmetrics = open(Metrics.txt, 'w+')
		fmodel.close()
		fmetrics.close()

	def augmentInput(self):
		#dbg('Input before augmenting: {}'.format(self.regressionData[x]))
		if self.types_of_nonlinearity[2]==1:
			self.regressionData[x] += [np.log(self.regressionData[x][0:self.no_original_ip])]
		if self.types_of_nonlinearity[1]==1:
			self.regressionData[x] += [np.exp(self.regressionData[x][0:self.no_original_ip])]
		if self.types_of_nonlinearity[0]==1:
			#Yet to implement
			return
		#dbg('Input after augmenting: {}'.format(self.regressionData[x]))

	def model_function_to_fit(self):
		pass
	