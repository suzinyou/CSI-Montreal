import csv
import numpy as np 
import os

names = ['aa_comp', 'hydrophobicity', 'polarity', 'polarizability', 'predicted_sec_struct', 'vdw_volume']
testLabels = np.load("../data/testing_labels.npy")


for filename in names:

    print("Loading %s and associated labels..." % filename)

    train = np.genfromtxt("../data/training/"+filename+"_training.csv", delimiter=",")
    train = train[ np.in1d(train[:,-1],testLabels), :]     # get only the examples with labels appearing in test set

    for l in testLabels:
        subset = train[train[:,-1]==l]
        if subset.shape[0] < 12:
       #     train = np.concatenate((train, np.concatenate((subset,subset))))
        #elif subset.shape[0] < 17:
            train = np.concatenate((train, subset))

    np.savetxt("../data/balanced_training/"+filename+".csv", train, delimiter=",")
    print ("Now you have "+str(train.shape[0])+" examples!!")

