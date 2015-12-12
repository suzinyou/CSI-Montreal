from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
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
#import BatchReader

# Some global vars for cross validation and number of parallel jobs to run
kfolds = 5
num_jobs = -1

print("Loading all files and associated labels...")

filenames = ['aa_comp', 'hydrophobicity', 'polarity', 'polarizability', 'predicted_sec_struct', 'vdw_volume']

testLabels = np.load("../data/testing_labels.npy")

train = np.genfromtxt("../data/balanced_training/aa_comp.csv", delimiter=",")
train = train[ np.in1d(train[:,-1],testLabels), :]     # get only the examples with labels appearing in test set
n_examples = train.shape[0]
examples = np.empty([n_examples,0])
labels = train[:,-1]

test = np.genfromtxt("../data/testing/aa_comp_testing.csv", delimiter=",")
n_test_examples = test.shape[0]
test_examples = np.empty([n_test_examples,0])
test_labels = test[:,-1]


for filename in filenames:
    subset_train = np.genfromtxt("../data/balanced_training/"+filename+".csv", delimiter=",")
    subset_train = subset_train[ np.in1d(subset_train[:,-1],testLabels), :]     # get only the examples with labels appearing in test set
    
    # concatenate these features to our data matrix (excluding labels column) 
    examples = np.concatenate((examples, subset_train[:,:-2]), axis=1)
    
    subset_test = np.genfromtxt("../data/testing/"+filename+"_testing.csv", delimiter=",")
    test_examples = np.concatenate((test_examples, subset_test[:,:-2]), axis=1)

    ### Classifiers and Transformers ###

print ("Number of features: %f" , examples.shape[1]) 

norm = Normalizer()
pca = PCA()
nb = MultinomialNB()
svc = SVC()

nusvc = NuSVC(verbose=True) # might need to add decision_function_shape='ovr' 


### Pipeline
pipeline=Pipeline(steps=[
# add transformer dictionary too
('norm', norm),
('pca', pca),
#('logit', logit),
#('nusvc', nusvc),
('nb', nb)
#('svc', svc)

])

### Parameters
parameters={
# parameters to perform grid-search over
'pca__n_components': [20, 40, 80],
'pca__whiten' : [True, False]
#'logit__solver' : ('liblinear','newton-cg'), 
#'logit__C': np.logspace(-6,6,4)
#'svc__C' : [1.0,10.0,100.0, 1000.0],
#'svc__gamma' : [1.0,10.0,100.0,1000.0]
#'svc__kernel' : ['rbf','sigmoid']
#'nusvc__nu': [1],    #An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
#'nusvc__tol': np.logspace(-3,3,3)   # penalty for error term (C=0.01) etc.
}

if __name__ == '__main__':
    grid_search = GridSearchCV(
        pipeline, parameters, verbose = 1, cv = kfolds, n_jobs = num_jobs)

    print("Starting grid search with the following pipeline and parameters")
    print("Pipeline:", [name for name, _ in pipeline.steps])
    print("Parameters:")
    date=str(datetime.now()).split(" ")[0]
    fulltime=str(datetime.now()).split(" ")
    realtime=fulltime[1].split(".")[0]
    #pickle_name=string.join((date, realtime), "--")
    pprint.pprint(parameters)
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
            best_parameters.append("\t%s:%s" % (param_name, best_params[param_name]))
    print best_parameters

    pickle_time = string.join((fulltime[0],realtime.replace(":","-")),"-")

    print "Accuracy score on test set: %0.3f" % (accuracy_score(test_labels, grid_search.predict(test_examples))) 
    train_scores.append(grid_search.best_score_)
    test_scores.append(accuracy_score(test_labels, grid_search.predict(test_examples)))



    with open("../data/pickle/Everything_NB.pkl", "w") as fp:
        cPickle.dump(grid_search, fp)

    print("Results are dumped as Everything_NB.pkl")
