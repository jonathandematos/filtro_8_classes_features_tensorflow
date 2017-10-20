#!/usr/bin/python
#
from __future__ import print_function
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import numpy as np
import joblib
import os, sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#
#
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
#
if(len(sys.argv) != 4):
    print("classify_paciente.py [pftas_file] [fold] [ampliacao]")
    exit(0)
#
ampliacao = int(sys.argv[3])
pftas_file = sys.argv[1]
fold = sys.argv[2]
print("Argumentos: {}".format(sys.argv))
#
# Combine results by vote
#
def CombineByVote(results):
    if(len(results)>0):
        if(len(results[0])>0):
            vote_list = [0 for i in range(len(results[0]))]
            for i in results:
                vote_list[np.argmax(np.array(i))] += 1
            return np.argmax(np.array(vote_list))
        return -1
    return -1
#
# Combine results by sum
#
def CombineBySum(results):
    if(len(results)>0):
        if(len(results[0])>0):
            vote_list = [0 for i in range(len(results[0]))]
            for i in results:
                for j in range(len(i)):
                    vote_list[j] += i[j]
            return np.argmax(np.array(vote_list))
        return -1
    return -1
#
#f = open("pftas_filtro_150.txt","r")
#f = open("svm_tissues/pftas_file_150.txt","r")
f = open(pftas_file, "r")
#
X = list()
Y = list()
Z = list()
W = list()
U = list()
errados = 0
for i in f:
    linha = i[:-1].split(";")
    a = linha[1].split("-")
    if(int(a[3]) == ampliacao):
        #Z.append(linha[1])
        W.append(str(a[0])+"-"+str(a[1])+"-"+str(a[2])+"-"+str(a[3])+"-"+str(a[4]))
        U.append(linha[1])
        x_tmp = list()
        for j in linha[2:-1]:
            x_tmp.append(float(j))
        if(len(x_tmp) != 162):
            errados += 1
            continue
        X.append(x_tmp)
        class_str = linha[0] #a[0].split("/")[8]#linha[0] #a[0].split("/")[8]#linha[0] #a[0].split("/")[8]
        #
        # 0 - importante
        # 1 - irrelevante
        #
        if(class_str == 'adenosis'):
        	class_line = int(0)
        if(class_str == 'ductal_carcinoma'):
        	class_line = int(4)
        if(class_str == 'fibroadenoma'):
        	class_line = int(1)
        if(class_str == 'lobular_carcinoma'):
        	class_line = int(5)
        if(class_str == 'mucinous_carcinoma'):
           	class_line = int(6)
        if(class_str == 'papillary_carcinoma'):
        	class_line = int(7)
        if(class_str == 'phyllodes_tumor'):
        	class_line = int(2)
        if(class_str == 'tubular_adenoma'):
        	class_line = int(3)
        Y.append(class_line)
#
f.close()
#print(errados)
##
##X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, test_size=0.3)
##
Z_test = list()
Z_train = list()
##
##f = open("svm_tissues/dsfold1.txt","r")
f = open(fold,"r")
#
for i in f:
    linha = i[:-1].split("|")
    if(int(linha[1]) == ampliacao):
        img = linha[0].split(".")[0]
        if(linha[3] == "train"):
            Z_train.append(img)
        else:
            Z_test.append(img)
f.close()
#
X_test = list()
Y_test = list()
U_test = list()
X_train = list()
Y_train = list()
U_train = list()
for i in range(len(X)):
    if(W[i] in Z_test):
        X_test.append(X[i])
        Y_test.append(Y[i])
        U_test.append(U[i])
    if(W[i] in Z_train):
        X_train.append(X[i])
        Y_train.append(Y[i])
        U_train.append(U[i])
#
#print(len(X_test), len(X_train))
#exit(0)
#
del X
del Y
del U
del Z_test
del Z_train
#for i in U_test:
#    print(i)
#exit(0)
#
#
#
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 10],
                     'C': [5e-1, 50, 5000, 50000]},
                    {'kernel': ['linear'], 'C': [1e-1, 1, 10]}]
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 5e-1],
#                     'C': [1, 100, 1000]}]
#
clf = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy', n_jobs=2)
#clf = SVC(probability=True)
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier(n_estimators=100)
#X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.30)
clf.fit(X_train, Y_train)
#
#print_parameters(clf)
print(clf.score(X_test, Y_test))
#
exit(0)
#
pacs = {}
imgs = {}
patch = {}
#
# SOB_B_A-14-22549G-100-030-550-150.png
#
correto = 0
total = 0
for i in range(len(X_test)):
    str_img = U_test[i].split("-")
    img = str(str_img[2])+str(str_img[3])+str(str_img[4])
    pac = str(str_img[0])+str(str_img[2])
    pred = np.squeeze(clf.predict_proba(np.array([X_test[i]])))
    print("{};{};".format(U_test[i], Y_test[i]), end="")
    for j in pred:
        print("{:.6f};".format(j), end="")
    print()
#    #
#    if(np.argmax(pred) == Y_test[i]):
#        correto += 1
#    total += 1
#    if(img in imgs):
#        imgs[img][1].append(pred)
#    else:
#        imgs[img] = [Y_test[i],list()]
#        imgs[img][1].append(pred)
#    #
#    if(pac in pacs):
#        pacs[pac][1].append(pred)
#    else:
#        pacs[pac] = [Y_test[i],list()]
#        pacs[pac][1].append(pred)
#
#joblib.dump(pacs, "pacs.pkl")
#joblib.dump(imgs, "imgs.pkl")
#
#pacs = joblib.load("pacs.pkl")
#imgs = joblib.load("imgs.pkl")
#print(pacs)
#
exit(0)
print("Patches: {}".format(float(correto)/total))
exit(0)
#
correto = 0
total = 0
for i in imgs:
    if(imgs[i][0] == CombineBySum(imgs[i][1])):
        correto += 1
    total += 1
print("Combinacao de imagens por soma: {}".format(float(correto)/total))
#
correto = 0
total = 0
for i in imgs:
    if(imgs[i][0] == CombineByVote(imgs[i][1])):
        correto += 1
    total += 1
print("Combinacao de imagens por voto: {}".format(float(correto)/total))
#
correto = 0
total = 0
for i in pacs:
    if(pacs[i][0] == CombineBySum(pacs[i][1])):
        correto += 1
    total += 1
print("Combinacao de paciente por soma: {}".format(float(correto)/total))
#
correto = 0
total = 0
for i in pacs:
    if(pacs[i][0] == CombineByVote(pacs[i][1])):
        correto += 1
    total += 1
print("Combinacao de paciente por voto: {}".format(float(correto)/total))
#
exit(0)
