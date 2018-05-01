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
import os

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
		self.createControlEqns()
		if os.path.exists('Models.txt'):
			os.remove('Models.txt')
		if os.path.exists('Metrics.txt'):
			os.remove('Metrics.txt')

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
		loss_table=[]
		state_loss_val=[]
		output_loss_val=[]
		regular=[]
		if name=='polynomial':
			self.predicted_model = np.polyfit(self.regressionData[x], self.regressionData[y], self.degree_of_polynomial)
		else:
			if name=='curve':
				self.predicted_model = sp.optimize.curve_fit(self.model_function_to_fit, self.regressionData[x], self.regressionData[y])
			else:
				if name=='ml_regression':
					state_model = skl.linear_model.LinearRegression(fit_intercept=False, copy_X=True)
					output_model = skl.linear_model.LinearRegression(fit_intercept=False, copy_X=True)
				else:
					if name=='robust':
						self.predicted_model = skl.linear_model.RANSACRegressor.fit(self.regressionData[x], self.regressionData[y]).get_params()
					else:
						if name=='l2':
							hyperparameter_optimization_time=time.time()
							for j in range(1,9):
								i=j/float(10)
								state_model = skl.linear_model.Ridge(alpha=i, fit_intercept=False, copy_X=True) 		#Tolerance ? Max iterations ?
								output_model = skl.linear_model.Ridge(alpha=i, fit_intercept=False, copy_X=True)
								state_model.fit(self.combinedIp0[:-len(self.combinedIp0)/5], self.regressionData[xdot][:-len(self.combinedIp0)/5])
								output_model.fit(self.combinedIp1[:-len(self.combinedIp0)/5], self.regressionData[y][:-len(self.combinedIp0)/5])
								state_loss = np.sum(np.power(np.subtract(state_model.predict(self.combinedIp0[-len(self.combinedIp0)/5:]), self.regressionData[xdot][-len(self.combinedIp0)/5:]), 2)) / (len(self.combinedIp0)/5)
								output_loss = np.sum(np.power(np.subtract(output_model.predict(self.combinedIp1[-len(self.combinedIp0)/5:]), self.regressionData[y][-len(self.combinedIp0)/5:]), 2)) / (len(self.combinedIp0)/5)
								loss_table += [(state_loss, output_loss, i)]
								state_loss_val += [state_loss]
								output_loss_val += [output_loss]
							regular += [loss_table[state_loss_val.index(max(state_loss_val))][2]]
							state_model = skl.linear_model.Ridge(alpha=regular[0], fit_intercept=False, copy_X=True)	
							regular += [loss_table[output_loss_val.index(max(output_loss_val))][2]]
							output_model = skl.linear_model.Ridge(alpha=regular[1], fit_intercept=False, copy_X=True)
							hyperparameter_optimization_time=time.time()-hyperparameter_optimization_time
							print('Regularization strengths used are {} and hyperparameter_optimization_time={}'.format(regular, hyperparameter_optimization_time))
						else:
							if name=='l1':
								hyperparameter_optimization_time=time.time()
								for j in range(1,9):
									i=j/float(10)
									state_model = skl.linear_model.Lasso(alpha=i, fit_intercept=False, copy_X=True) 		#Tolerance ? Max iterations ?
									output_model = skl.linear_model.Lasso(alpha=i, fit_intercept=False, copy_X=True)
									state_model.fit(self.combinedIp0[:-len(self.combinedIp0)/5], self.regressionData[xdot][:-len(self.combinedIp0)/5])
									output_model.fit(self.combinedIp1[:-len(self.combinedIp0)/5], self.regressionData[y][:-len(self.combinedIp0)/5])
									state_loss = np.sum(np.power(np.subtract(state_model.predict(self.combinedIp0[-len(self.combinedIp0)/5:]), self.regressionData[xdot][-len(self.combinedIp0)/5:]), 2)) / (len(self.combinedIp0)/5)
									output_loss = np.sum(np.power(np.subtract(output_model.predict(self.combinedIp1[-len(self.combinedIp0)/5:]), self.regressionData[y][-len(self.combinedIp0)/5:]), 2)) / (len(self.combinedIp0)/5)
									loss_table += [(state_loss, output_loss, i)]
									state_loss_val += [state_loss]
									output_loss_val += [output_loss]
								regular += [loss_table[state_loss_val.index(max(state_loss_val))][2]]
								state_model = skl.linear_model.Lasso(alpha=regular[0], fit_intercept=False, copy_X=True)	
								regular += [loss_table[output_loss_val.index(max(output_loss_val))][2]]
								output_model = skl.linear_model.Lasso(alpha=regular[1], fit_intercept=False, copy_X=True)
								hyperparameter_optimization_time=time.time()-hyperparameter_optimization_time
								print('Regularization strengths used are {} and hyperparameter_optimization_time={}'.format(regular, hyperparameter_optimization_time))
							else:
								sys.exit('Invalid solver option')
		
		state_model.fit(self.combinedIp0, self.regressionData[xdot])
		output_model.fit(self.combinedIp1, self.regressionData[y])
		combined_ip_model = self.array2list(state_model.coef_)
		combined_op_model = self.array2list(output_model.coef_)
		self.splitCombinedModels(combined_ip_model, combined_op_model)
		self.model_prediction_time_secs = time.time()-start
		#dbg('Model predicted as {} in time {} seconds'.format(self.predicted_model, self.model_prediction_time_secs))
		return(self.model_prediction_time_secs, self.predicted_model)

	def array2list(self, ar):
		temp=copy.deepcopy(ar)
		for i in range(len(ar)):
			temp[i] = ar[i].tolist()
		return temp

	def getOutput(self):
		fmodel = open('Models.txt', 'a')
		#fmetrics = open('Metrics.txt', 'w+')
		for mat in self.predicted_model:
			for i in range(len(mat)):
				for entry in mat[i]:
					fmodel.write(str(entry)+' ')
				fmodel.write('\n')
			fmodel.write('\n')
		fmodel.write('\n---------------------------------------------------------------------------------------------\n')
		fmodel.close()
		#fmetrics.close()
	