#!/usr/bin/env python3
import copy
import sys
import numpy as np
import scipy as sp 
import sklearn as skl
from sklearn import linear_model
import time
import itertools as it
import matplotlib.pyplot as plt

debug=1
u=0 #Inputs
x=1 #
xdot=2
y=3
A=0
B=1
C=2
D=3

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
	form=[]

	def __init__(self, matlab_api=None, input_data_file=None, degree=1, regularization=0, data=None, form=None):
		if input_data_file != None:
			self.readFile(input_data_file)
		else:
			if data!=None:
				self.regressionData=[]
				self.regressionData=data
				for j in range(len(data[0])):
					self.regressionData[u][j] = [float(i) for i in data[u][j]]
					self.regressionData[x][j] = [float(i) for i in data[x][j]]
					self.regressionData[xdot][j] = [float(i) for i in data[xdot][j]]
					self.regressionData[y][j] = [float(i) for i in data[y][j]]
		self.form=form
		if self.regressionData==None:
			sys.exit('No input data found')
		self.degree_of_polynomial=degree
		self.regularization_strength=regularization
		if self.degree_of_polynomial!=1:
			self.augmentInput()
		self.createControlEqns()
		
	def createControlEqns(self):
		self.combinedIp0=copy.deepcopy(self.regressionData[x])
		self.combinedIp1=copy.deepcopy(self.regressionData[x])
		for i in range(len(self.combinedIp0)):
			self.combinedIp0[i] += copy.deepcopy(self.regressionData[u][i])
			self.combinedIp1[i] += copy.deepcopy(self.regressionData[u][i])
			if self.degree_of_polynomial>1:
				self.combinedIp0[i] += copy.deepcopy([k**self.degree_of_polynomial for k in self.regressionData[x][i]])
		dbg('Combined control eqns: x:{} u:{} combined-state{}'.format(self.regressionData[x][-1], self.regressionData[u][-1], self.combinedIp0[-1]))
	

	def visualize(self):
		plt.figure(1)
		plt.subplot(211)
		plt.plot(self.regressionData[u])
		plt.subplot(212)
		plt.plot(self.regressionData[y])
		plt.show()		
		plt.figure(2)
		plt.subplot(311)
		plt.plot(self.regressionData[x])
		plt.subplot(312)
		plt.plot(self.regressionData[xdot])
		plt.show()

	def readFile(self, fname):
		fhandle = open(fname, 'r')
		temp = fhandle.readlines()[1:]
		dataSet=[]
		for t in temp:
			dataSet += [t.strip('\n').split()[0:5]]
		fhandle.close()

	def splitCombinedModels(self, ip, op):
		Amat=[]
		Bmat=[]
		Cmat=[]
		Dmat=[]
		Emat=[]
		dbg('ip model : {}\nop_model:{}'.format(ip, op))
		for i in range(len(ip)):
			Amat += [copy.deepcopy(ip[i][:self.form[x]]).tolist()]
			Bmat += [copy.deepcopy(ip[i][self.form[x]:self.form[x]+self.form[u]]).tolist()]
			if self.degree_of_polynomial>1:
				Emat += [copy.deepcopy(ip[i][-self.form[x]:]).tolist()]
		dbg('A:{}\n\nB:{}\n\nE:{}\n'.format(Amat,Bmat,Emat))
		for i in range(len(op)):
			Cmat += [copy.deepcopy(op[i][:self.form[x]]).tolist()]
			Dmat += [copy.deepcopy(op[i][self.form[x]:]).tolist()]
		dbg('C:{}\n\nD:{}'.format(Cmat,Dmat))
		self.predicted_model=[Amat,Bmat,Cmat,Dmat]
		if self.degree_of_polynomial>1:
			self.predicted_model.append(Emat)

	def solver(self, name): #polynomial, curve, ml_regression, robust
		start = time.time()
		if name=='polynomial':
			self.predicted_model = np.polyfit(self.regressionData[x], self.regressionData[y], self.degree_of_polynomial)
		else:
			if name=='curve':
				self.predicted_model = sp.optimize.curve_fit(self.model_function_to_fit, self.regressionData[x], self.regressionData[y])
			else:
				if name=='ml_regression':
					state_model = skl.linear_model.LinearRegression(fit_intercept=False)
					state_model.fit(self.combinedIp0 ,self.regressionData[xdot])
					combined_ip_model = self.array2list(state_model.coef_)
					output_model = skl.linear_model.LinearRegression(fit_intercept=False)
					output_model.fit(self.combinedIp1 ,self.regressionData[y])
					combined_op_model = self.array2list(output_model.coef_)
					self.splitCombinedModels(combined_ip_model, combined_op_model)
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
		#dbg('Model predicted as {} in time {} seconds'.format(self.predicted_model, self.model_prediction_time_secs))
		return(self.model_prediction_time_secs, self.predicted_model)

	def array2list(self, ar):
		temp=copy.deepcopy(ar)
		for i in range(len(ar)):
			temp[i] = ar[i].tolist()
		return temp

	def getOutput(self):
		fmodel = open('Model.txt', 'w+')
		#fmetrics = open('Metrics.txt', 'w+')
		for mat in self.predicted_model:
			for i in range(len(mat)):
				for entry in mat[i]:
					fmodel.write(str(entry)+' ')
				fmodel.write('\n')
			fmodel.write('\n')
		fmodel.close()
		#fmetrics.close()

	def augmentInput(self):
		pass

	def model_function_to_fit(self):
		pass
	