# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:20:48 2015

@author: Suzin
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
#from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import ShuffleSplit
#from String import find



class plotter ():
    def __init__ (self, filename, fullname):
        self.filename = filename
        self.fullname = fullname

        # Loading the first 20000 as the training set and third 10000 as the test set.
        train = np.genfromtxt("../data/training/"+filename+"_training.csv", delimiter=",")
        test = np.genfromtxt("../data/testing/"+filename+"_testing.csv", delimiter=",")
        self.testLabels = np.load("../data/testing_labels.npy")
        train = train[ np.in1d(train[:,-1],testLabels), :]     # get only the examples with labels appearing in test set
        #np.random.shuffle(data)

        self.Xtrain = train[:,:-2]
        self.ytrain = train[:,-1]
        self.Xtest = test[:,:-2]
        self.ytest = test[:,-1]

    def plot_confusion_matrix(c, t): 
        # ...(cm, cmap=plt.cm.Blues)
        plt.imshow(c, interpolation='nearest')
        #plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(t))
        plt.xticks(tick_marks, t)
        plt.yticks(tick_marks, t)
        plt.ylabel('True fold')
        plt.xlabel('Predicted fold')
        plt.tight_layout()

    # plotting confusion matrix: only call on test set
    def cm(self):
        dill = cPickle.load(open("../data/pickle/"+self.fullname+".pkl"))

        # plt.rcdefaults() 
        # do not call plt.xkcd()...
        plt.figure(1)
        ypred = dill.predict(self.Xtest)
        cm = confusion_matrix(self.ytest, y_pred)
        plot_confusion_matrix(cm, self.testLabels)
        plt.savefig("../figures/cm_"+self.fullname+".png", dpi=300)


    '''
    plt.figure(2)
    y_pred = dill.predict_proba(X_test)
    y_score = np.max(y_pred, axis=1)
    fpr, tpr, th = roc_curve(y_test, y_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr)
    plt.savefig("../figures/roc_logreg_pca.png", dpi=300)
    '''

    # Plotting n_components v's explained_variance_ for PCA
    def pca_var(self):
        dill = cPickle.load(open("../data/pickle/"+self.fullname+".pkl"))
        params = dill.best_estimator_.get_params()
        #pca_params = dict((k, params[k]) for k in ('pca__copy', 'pca__n_components', 'pca__whiten'))
        pca = PCA(copy=params['pca__copy'], n_components=params['pca__n_components'], whiten=params['pca__whiten'])
        pca.fit(self.Xtrain)
        plt.figure()
        plt.clf()
        plt.axes([0.2, .2, 0.7, 0.7])
        plt.plot(pca.explained_variance_, linewidth=2)
        plt.xlabel('n_components')    
        plt.axis('tight')
        plt.ylabel('explained_variance_')
        
        ###############################################################################
        # Prediction
        #plt.axvline(dill.best_estimator_.named_steps['pca'].n_components,
    #                linestyle=':', label='n_components chosen')
        #plt.legend(prop=dict(size=12))
        plt.savefig("../figures/pcavar_"+self.fullname+".png", dpi=300)


    # plotting learning curves: only call on training sets
    def l_curve(self):
        dill = cPickle.load(open("../data/pickle/"+self.fullname+".pkl"))
        
        cv = ShuffleSplit(len(y), n_iter=5, test_size=0.2, random_state=0)
        train_sizes, train_scores, test_scores = learning_curve(dill.best_estimator_, self.Xtrain, self.ytrain, cv=cv, train_sizes=np.linspace(.1, 1.0, 5))
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.savefig("../figures/lc_"+self.fullname+".png", dpi=300)