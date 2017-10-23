#!/usr/bin/python
#
from __future__ import print_function
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import joblib
import os, sys
#
def print_parameters(clf):
    print("Best parameters with score {:.5f}% set found on development set:".format(clf.best_score_))
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#############################
# LOADING CRC
#############################
f = open("crc_features.txt","r")
#
X = list()
Y = list()
t = 0
for i in f:
	linha = i[:-1].split(";")
        #print(linha[0][0:2])
	c = int(linha[0][0:2])
        t += 1
        if(c >= 1 and c<=8):
            x_tmp = list()
            for j in linha[2:-1]:
                x_tmp.append(float(j))
            if(len(x_tmp) != 4096):
                continue
            X.append(x_tmp)
            #
            # 0 - importante
            # 1 - irrelevante
            #
            if(c > 2):
                Y.append(1)
            else:
                Y.append(0)
#
f.close()
#############################
# TRAINING SVM
#############################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
#
del X
del Y
#
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 5e-1],
                     'C': [5e-1, 1, 5]},
                    {'kernel': ['linear'], 'C': [1e-1, 1, 10]}]
#
if( os.path.exists("classificador_crc.pkl") == True ):
    clf = joblib.load("classificador_crc.pkl")
else:
    #
    #clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy', n_jobs=2, verbose=5)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=2)
    clf.fit(X_train, Y_train)
    #
    #print_parameters(clf)
    #
    #clf = SVC(probability=True)
    #clf.fit(X_train, Y_train)
    joblib.dump(clf, "classificador_crc.pkl")
#
#print_parameters(clf)
print(clf.score(X_test, Y_test))
del X_train
del X_test
del Y_train
del Y_test
#
#############################
# LOADING BREAKHIS
#############################
#
f = open(sys.argv[1],"r")
#
X = list()
Y = list()
Z = list()
p = 0
for i in f:
    #if(p == 0):
    #    p = 1
    #    continue
    linha = i[:-1].split(";")
    x_tmp = list()
    for j in linha[1:-1]:
        x_tmp.append(float(j))
    if(len(x_tmp) != 4096):
        print("erro")
        continue
    X.append(x_tmp)
    #class_str = linha[0] #linha[0].split("/")[8]
    class_str = linha[0].split("-")[0].split("_")[2]
    #
    # 0 - importante
    # 1 - irrelevante
    #
    if(class_str == 'A'):
    	class_line = int(0)
    if(class_str == 'DC'):
    	class_line = int(4)
    if(class_str == 'F'):
    	class_line = int(1)
    if(class_str == 'LC'):
    	class_line = int(5)
    if(class_str == 'MC'):
    	class_line = int(6)
    if(class_str == 'PC'):
    	class_line = int(7)
    if(class_str == 'PT'):
    	class_line = int(2)
    if(class_str == 'TA'):
	class_line = int(3)
    #
    Y.append(class_str)
    Z.append(linha[0])
#
f.close()
#for i in Y:
#    print(i)
#
f = open(sys.argv[2],"w")
for i in range(len(X)):
    pred = clf.predict_proba(np.array([X[i]]))
    if(pred.argmax() == 0):
        f.write("{};{};".format(Y[i], Z[i]))
        for j in X[i]:
            f.write("{:.6f};".format(j))
        f.write("\n")
#
f.close()
exit(0)
