#!/usr/bin/env python3
import sys
import math
import csv
import re
from collections import Counter
from matplotlib import pyplot as plt
#import numpy as np

def sigmoid(x):
    return 1/(1+math.exp(-x))

def dot_product (X,Theta):
    product = 0.0
    for idx,val in X.iteritems():
        product += (float(val)*Theta[idx])
    return product

def compute_gradient (Y_i,X_i,Theta,M):
    gradient_i = {}
    z = dot_product(X_i,Theta)
    sgmd = sigmoid(z)
    h = Y_i -sgmd
    for (idx,Xm_i) in X_i.iteritems():
        gradient_i[idx] = h*float(Xm_i)
    return gradient_i

def calc_NLL(Y,X,Theta):

    J = float(0.0)
    for i in range(len(X)):
        X_i = X[i]
        Y_i = Y[i]    
        z = dot_product(X_i,Theta)
        LL = (-Y_i)*(z)+math.log(1+math.exp(z))
        J += LL
    #print ("NLL is: ",J)
    return J/len(X)

def predict(Y,X,Theta,out_file,metrics_file,mode):
    correct = 0.0
    mistake = 0.0
    for i in range(len(X)):
        prediction = -1
        z = dot_product(X[i],Theta)
        sgmd = sigmoid(z)
        #print (i,"sigmoid: ",sgmd)
        #print (i,"Actual label: ",Y[i])
        if (sgmd >= 0.5):
            prediction = 1
        else:
            prediction = 0
        if (Y[i] == prediction):
            correct += 1
        else:
            mistake += 1
        if (mode != "validate"):
            out_file.write(str(prediction)+'\n')
    #print ("correct: ",str(correct))
    #print ("mistake: ",str(mistake))
    total = correct + mistake
    error = mistake/(float(total))
    #print ("error: ",str(error))
    if mode != "validate":
        metrics_file.write("error("+mode+"): "+str(error)+"\n")
        
def SGD (Y,X,Theta0,epochs):
    #Step 1
    #Choose initial point 
    Theta = list(Theta0)

    #Step 2
    #While not converged, here convergence criteria is the number of epochs run
    for epoch in range(epochs):
        #Iterate over each example
        for i in range(len(X)):
            #Step 2.a
            #Compute gradients
            M = len(X[i])
            gradient_i = compute_gradient(Y[i] ,X[i] ,Theta,M)

            #Step 2.b
            #Choose learning rate (learning rate is constant here)
            Eta = 0.1

            #Step 2.c
            #update Theta on each of the examples
            for idx in X[i].keys():
                Theta[idx] += Eta*gradient_i[idx]
        J = calc_NLL(Y,X,Theta) 
    #Step 3
    #Return updated parameter vector after 
    return Theta

def SGD (Y,X,Theta0,epochs):
    #Step 1
    #Choose initial point 
    Theta = list(Theta0)

    #Step 2
    #While not converged, here convergence criteria is the number of epochs run
    for epoch in range(epochs):
        #Iterate over each example
        for i in range(len(X)):
            #Step 2.a
            #Compute gradients
            M = len(X[i])
            gradient_i = compute_gradient(Y[i] ,X[i] ,Theta,M)

            #Step 2.b
            #Choose learning rate (learning rate is constant here)
            Eta = 0.1

            #Step 2.c
            #update Theta on each of the examples
            for idx in X[i].keys():
                Theta[idx] += Eta*gradient_i[idx]
        J = calc_NLL(Y,X,Theta) 
    #Step 3
    #Return updated parameter vector after 
    return Theta

def SGD_plot (Y_train,X_train,Y_valid,X_valid,Theta0,epochs): 
    #Step 1
    #Choose initial point 
    Theta = list(Theta0)

    train_NLLS = []
    valid_NLLS = []

    #Step 2
    #While not converged, here convergence criteria is the number of epochs run
    for epoch in range(epochs):
        #Iterate over each example
        for i in range(len(X_train)):
            #Step 2.a
            #Compute gradients
            M = len(X_train[i])
            gradient_i = compute_gradient(Y_train[i] ,X_train[i] ,Theta,M)

            #Step 2.b
            #Choose learning rate (learning rate is constant here)
            Eta = 0.1

            #Step 2.c
            #update Theta on each of the examples
            for idx in X_train[i].keys():
                Theta[idx] += Eta*gradient_i[idx]
        J = calc_NLL(Y_train,X_train,Theta)
        train_NLLS.append(J)
        J = calc_NLL(Y_valid,X_valid,Theta)
        valid_NLLS.append(J)
    #print (len(train_NLLS) == epochs)
    #for elem in train_NLLS:
    #    print elem
    print ("plotting the graph.")
    x_axis = [epoch+1 for epoch in range(epochs)]
    plt.plot(x_axis,train_NLLS,'b',label='training NLL')
    plt.plot(x_axis,valid_NLLS,'g',label='validation NLL')
    plt.ylabel("Negative Log Likelihood (NLL)")
    plt.xlabel("Epochs")
    plt.axis([-1,epochs+10,-1,6])
    plt.legend(loc = 'upper right', shadow = True, fontsize = 'x-large' )
    plt.show()

    #Plot train NLLs and validation NLLs against epochs.

def get_data (raw_data):
    data = csv.reader(raw_data,delimiter='\t')
    Y = []
    X = []
    i = 1 
    for row in data:
        Y.append(int(row[0]))
        #print (row[0])
        X_i = {}
        #xo for any vector X(i) is 1, since it is multiplied with THETA0 which is the bias term
        X_i[0] = 1
        for pairing in row[1:]:
            p = pairing.split(":")
            #print p
            #shift indices of all other terms to account for the bias term
            idx = int(p[0])+1
            value = float(p[1])
            X_i[idx] = value
        X.append(X_i)
    return Y,X
    

train_in_file_name = sys.argv[1]
validation_in_file_name = sys.argv[2]
test_in_file_name = sys.argv[3]
dict_in_file_name = sys.argv[4]
train_out_file_name = sys.argv[5] 
test_out_file_name =sys.argv[6]
metrics_out_file_name = sys.argv[7]
epochs = int(sys.argv[8])

metrics_out = open(metrics_out_file_name,'w+')

V = len(open(dict_in_file_name).readlines())

#print V
Theta = [float(0.0) for i in range(V+1)]

#Training phase
mode = "train"
train_in = open(train_in_file_name,'r')
train_out = open(train_out_file_name,'w+')
Y,X = get_data(train_in)
Y_train = list(Y)
X_train = list(X)
#Theta = SGD(Y,X,Theta,epochs)
#predict(Y,X,Theta,train_out,metrics_out,mode)
train_in.close()

#Validation phase
mode = "validate"
valid_in = open(validation_in_file_name,'r')
Y,X = get_data(valid_in)
Y_valid = list(Y)
X_valid = list(X)
#predict(Y,X,Theta,None,None,mode)
valid_in.close()

#Testing phase
mode = "test"
test_in  = open(test_in_file_name,'r')
test_out  = open(test_out_file_name,'w+')
Y,X = get_data(test_in)
Y_test = list(Y)
X_test = list(X)
#predict(Y,X,Theta,test_out,metrics_out,mode)
test_in.close()

Theta = [float(0.0) for i in range(V+1)]

SGD_plot(Y_train, X_train, Y_valid, X_valid, Theta,epochs)

mode = "train"
predict(Y_train,X_train,Theta,train_out,metrics_out,mode)
mode = "test"
predict(Y_test,X_test,Theta,test_out,metrics_out,mode)
