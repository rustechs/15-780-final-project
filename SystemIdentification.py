#!/usr/bin/env python3

import numpy as np
import scipy as sp 
import sklearn as skl
import time

debug=1

def dbg(msg):
	if debug==1:
		print(msg)
	return

class SI():
	regressionData=None
	x=0 #Inputs
	y=1 #Labels
	predicted_model=None
	model_prediction_time_secs=None

	def __init__(self, matlab_api=None, input_data_file=None):
		if input_data_file not None:
			self.readFile(input_data_file)
		if regressionData==None:
			sys.exit('No input data found')

	def readFile(self, fname):
		#Get format from Rishabh
		fhandle = open(fname, 'r')
		fhanfle.close()


	def solver(self. name):
		start = time.time()
		if name=='polynomial':
			np.polyfit(self.regressionData[x], self.regressionData[y], degree_of_polynomial)
		else if name=='curve':
			sp.optimize.curve_fit(model_function_to_fit, self.regressionData[x], self.regressionData[y])
		else if name=='ml_regression':
			skl.linear_model.LinearRegression.fit(self.regressionData[x], self.regressionData[y], list_of_sample_weights).get_params()
		else:
			sys.exit('Invalid solver option')

		model_prediction_time_secs = time.time()-start
		dbg('Model predicted as {} in time {} seconds'.format(self.predicted_model, self.model_prediction_time_secs))
		return(self.model_prediction_time_secs, self.predicted_model)

	def getOutput(self):
		#write to file or output to api
		fmodel = open(Model.txt, 'w+')
		fmetrics = open(Metrics.txt, 'w+')
		fmodel.close()
		fmetrics.close()