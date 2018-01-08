#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
__title__ = 'PLA and Pocket algo'
__author__ = 'Xinru'
__mtime__ = '2017/12/21'
'''

from __future__ import division
import random
import string
from numpy import *
import numpy as np
from numpy.linalg import *

def Data_pretreatment(path):
    rawData = open(path).readlines()
    dataNum = len(rawData)
    dataDim = len(rawData[0].strip().split(' ')) + 1
    dataIdx = 0
    X = zeros([dataNum,dataDim])
    X[:,0] = 1
    Y =zeros(dataNum)
    for line in rawData:
        temp = line.strip().split('\t')
        temp[0] = temp[0].split(' ')
        X[dataIdx,1:] = temp[0]
        Y[dataIdx]=temp[1]
        dataIdx = dataIdx + 1
    return X,Y

def is_same_sign(W,vector,y):
    if (dot(W,vector)>0 and y>0) or (dot(W,vector) < 0 and y <0):
        return 1
    else:
        return -1

def n_error(W,X,Y):
    m = len(X)
    N = 0
    for i in range(m):
        if is_same_sign(W,X[i],Y[i]) == -1:
            N = N + 1
    return N
            
    
# N0.15
'''
def PLA_naive(X,Y,iterations):
    (m,n) = X.shape
    W = zeros([n])
    update = 0
    last_update = 0
    for i in range(iterations):
        for j in range(m):
            if  is_same_sign(W,X[j],Y[j])== -1:
                W = W + Y[j]* X[j]
                update = update + 1
                last_update = j
    return W,update,last_update'''

#N0.16
'''
def PLA_random(X,Y,iterations,eta):
    (m,n) = X.shape
    W = zeros([n])
    update = 0
    for i in range(iterations):
        sample = random.choice(range(m))
        if  is_same_sign(W,X[sample],Y[sample])== -1:
            W = W + eta*Y[sample]* X[sample]
            update = update + 1
    return W,update  '''

#No.17
def PLA_pocket(X,Y,iterations,eta):
    (m,n) = X.shape
    update = 0
    W_p = zeros([n])
    W = zeros([n])
    N_p = n_error(W_p,X,Y)
    while update <=iterations:
        sample = random.choice(range(m))
        if is_same_sign(W,X[sample],Y[sample]) == -1:
            update = update + 1
            W = W + eta * Y[sample] *X[sample]
            N = n_error(W,X,Y)
            if N < N_p:
                W_p= W
                N_p = N
                #print('N_error_try=',N_error_try)
    return W_p

'''def PLA_pocket(X, Y, iterateTimes,eta):  
    (dataNum, dataDim) = X.shape  
    W_p = zeros(dataDim)  
    id_p = 0  
    ErrNum_p = n_error(W_p,X,Y)  
    W = zeros(dataDim)  
    ErrNum = 0  
    iterate = 0  
    dataIdx = 0  
    #random.seed(int(time.time() % 3000))  
    while iterate <= iterateTimes:  
        dataIdx = random.choice(range(dataNum))  
        if is_same_sign(W, X[dataIdx], Y[dataIdx]) == -1:  
            iterate += 1  
            W = W + eta * Y[dataIdx] * X[dataIdx] #!!!! W += eta * Y[dataIdx] * X[dataIdx] will change value of W_p  
            ErrNum = n_error(W,X,Y)  
            if ErrNum < ErrNum_p:  
                W_p = W  
                ErrNum_p = ErrNum  
    # print 'ErrNum_p: ' + str(ErrNum_p)  
    return W_p '''

#No.18
'''
def PLA_pocket2(X,Y,iterations,eta):
    (m,n) = X.shape
    update = 0
    W = zeros([n])
    while update <=iterations:
        sample = random.choice(range(m))
        if is_same_sign(W,X[sample],Y[sample])==-1:
            update = update + 1
            W = W + eta * Y[sample] *X[sample]
    return W'''


        
    
        
if __name__ == '__main__':
    X_train, Y_train = Data_pretreatment('hw1_18_train.dat')
    X_test,Y_test = Data_pretreatment('hw1_18_test.dat')
    Xm_train = mat(X_train)
    Ym_train = mat(Y_train)
    X = np.hstack((Xm_train,Ym_train.T))
    
    iterations = 50
    Experiments = 100
    eta = 0.5
    total_error = 0
    m = len(X_test)
    for i in range(Experiments):
        random.shuffle(X)
        X_train = X[:,:5]
        Y_train = X[:,5]
        X_train = array(X_train)
        Y_train = array(Y_train)
        W = PLA_pocket(X_train,Y_train,iterations,eta)
        N_error = n_error(W,X_test,Y_test)
        total_error = total_error + N_error
    print(total_error/m/Experiments)





    
