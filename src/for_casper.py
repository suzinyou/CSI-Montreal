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

from feature_set import get_all_feature_sets

# Some global vars for cross validation and number of parallel jobs to run
kfolds = 5
num_jobs = -1

filenames = ['aa_comp', 'hydrophobicity', 'polarity', 'polarizability', 'predicted_sec_struct', 'vdw_volume']

train_scores = []
test_scores = []

setts = get_all_feature_sets()

for key in setts:

    # print("Loading %s and associated labels..." % filename)

    # train = np.genfromtxt("../data/training/"+filename+"_training.csv", delimiter=",")
    # testLabels = np.load("../data/testing_labels.npy")
    # train = train[ np.in1d(train[:,-1],testLabels), :]     # get only the examples with labels appearing in test set
    # #np.random.shuffle(data)
    # examples = train[:,:-2]
    # labels = train[:,-1]

    # test = np.genfromtxt("../data/testing/"+filename+"_testing.csv", delimiter=",")
    # test_examples = test[:,:-2]
    # test_labels = test[:,-1]

    examples = setts[key]['train_X']
    labels = setts[key]['train_Y']
    test_examples = setts[key]['test_X']
    test_labels = setts[key]['test_Y']

    print "Using the feature set of "+key


    n_examples, n_features = examples.shape

    print("Loaded "+str(n_examples)+" examples.")

    ### Classifiers and Transformers ###
    norm = Normalizer()
    poly = PolynomialFeatures()
    pca = PCA(n_components=n_features, whiten=True)
    logit = LogisticRegression(solver='newton-cg',multi_class='multinomial')
    svc = SVC(kernel='sigmoid')
    nusvc = NuSVC(verbose=True) # might need to add decision_function_shape='ovr' 


    ### Pipeline
    pipeline=Pipeline(steps=[
        # add transformer dictionary too
        ('poly', poly),
        ('norm', norm),
        #('pca', pca),
        #('logit', logit),
        #('nusvc', nusvc),
        ('svc', svc)
    ])

    ### Parameters
    parameters={
        # parameters to perform grid-search over
        'poly__degree': [1,2],
        #'pca__n_components': [16, 64, 256, 576],
        #'logit__solver' : ('liblinear','newton-cg'), 
        #'logit__C': np.logspace(-6,6,4)
        'svc__C' : np.logspace(-2, 10, 13),
        'svc__gamma' : np.logspace(-9, 3, 13)
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
            print("\t%s: %r" % (param_name, best_params[param_name]))

        pickle_time = string.join((fulltime[0],realtime.replace(":","-")),"-")

        print "Accuracy score on test set: %0.3f" % (accuracy_score(test_labels, grid_search.predict(test_examples))) 
        train_scores.append(grid_search.best_score_)
        test_scores.append(accuracy_score(test_labels, grid_search.predict(test_examples)))

      
        for key in parameters:
            filename = filename+"_"+key

        with open("../data/pickle/"+filename+"_"+pickle_time+".pkl", "w") as fp:
            cPickle.dump(grid_search, fp)
            print("Results are dumped as "+filename+"_"+pickle_time+".pkl")
        
data = zip(filenames, train_scores, test_scores)
print "Feature Set     CV SCORE      test SCORE"
for d in data:
    print d
