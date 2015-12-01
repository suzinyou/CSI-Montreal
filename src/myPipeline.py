from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import accuracy_score

import pprint as pprint
import os
import numpy as np
import scipy.sparse as sps
import string
from collections import Counter
from datetime import datetime
from time import time
import cPickle

class myPipeline ():

	parameters={
	# parameters to perform grid-search over
	'poly__degree': [1,2],
	'pca__n_components': [16, 64, 256, 576],
	'logit__solver' : ('liblinear','newton-cg'), 
	'logit__C': np.logspace(-6,6,4),
	'svc__C' : np.logspace(-2, 10, 13),
    'svc__gamma' : np.logspace(-9, 3, 13)
    #'nusvc__nu': [1],    #An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    #'nusvc__tol': np.logspace(-3,3,3)   # penalty for error term (C=0.01) etc.
	}

	def __init__ (self, folds=5, num_jobs=-1, filename='aa_comp', preprocess=['poly','norm'], clf='svc'):
		self.folds = folds
		self.num_jobs = num_jobs
		self.filename = filename			# list of files to load -- for now I will just use one file at a time.
		self.preprocess = preprocess 		# list of strings of which preprocessing to do
		self.clf = clf

	def instantiate (self, n_features):
		# Classifiers
		norm = Normalizer()
		poly = PolynomialFeatures()
		pca = PCA(n_components=n_features, whiten=True)
		logit = LogisticRegression(solver='newton-cg',multi_class='multinomial')
		svc = SVC()
		nusvc = NuSVC(verbose=True) # might need to add decision_function_shape='ovr' 
		
		if 'pca' in self.preprocess and 'svc'==self.clf:
			self.pipeline = Pipeline(steps=[('poly', poly),('norm', norm),('pca', pca),('svc', svc)])
			self.parameters = {k: parameters[k] for k in self.preprocess.append(self.clf)}
		else: 
			self.pipeline = Pipeline(steps=[('poly',poly),('norm', norm),('pca', pca),('svc', svc)])

	def load (self):
		print("Loading %s and associated labels..." % self.filename)

		data = np.genfromtxt("../data/training/"+self.filename+"_training.csv", delimiter=",")
		testLabels = np.load("../data/testing_labels.npy")
		data = data[ np.in1d(data[:,-1],testLabels), :] 	# get only the examples with labels appearing in test set

		return data[:,:-2], data[:,-1]

	def getResults (self):
		examples, labels = self.load()
		n_examples, n_features = examples.shape
		print("Loaded "+str(n_examples)+" examples.")

		self.instantiate(n_features)

		grid_search = GridSearchCV(self.pipeline, self.parameters, verbose = 1, cv = self.folds, n_jobs = self.num_jobs)

		print("Starting grid search with the following pipeline and parameters")
		print("Pipeline:", [name for name, _ in self.pipeline.steps])
		print("Parameters:")
		date=str(datetime.now()).split(" ")[0]
		fulltime=str(datetime.now()).split(" ")[1]
		realtime=fulltime.split(".")[0]
		#pickle_name=string.join((date, realtime), "--")
		pprint.pprint(self.parameters)
		t0=time()
		grid_search.fit(examples, labels)
		print("Done in %0.3fs" % (time() - t0))
		print()

		print("Best score: %0.3f" % grid_search.best_score_)
		print("Estimator: ")
		pprint.pprint(grid_search.best_estimator_)
		print("Best results obtained with the following parameters:")
		best_params=grid_search.best_estimator_.get_params()
		for param_name in (sorted(parameters.keys())):
			print("\t%s: %r" % (param_name, best_params[param_name]))

		f = self.filename
		for key in self.parameters:
			f = f+key+"_"

		with open("../data/pickle/"+f+string.rstrip(str(time()), ".")+".pkl", "w") as fp:
			cPickle.dump(grid_search, fp)
    
		print("Results are dumped as"+f+string.rstrip(str(time()), ".")+".pkl")