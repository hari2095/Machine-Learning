#!/usr/bin/env python
import sys
import numpy as np
#from numpy.lib.recfunctions import append_fields

class intermediates:
    def __init__(self,x,a,z,b,y_hat,J):
        self.x = x
        self.a = a
        self.z = z
        self.b = b
        self.y_hat = y_hat
        self.J = J

def sigmoid(a):
    return 1/(1+np.exp(-a))

def init_weights(D,K,M):
    #Zero weights for the bias terms
    Alpha = np.zeros((D,1))
    Beta  = np.zeros((K,1))
    if(RANDOM == init_flag):
        random_alphas = np.random.uniform(-0.1,0.1,(D,M))
        Alpha = np.append(Alpha,random_alphas,axis=1)
        random_betas  = np.random.uniform(-0.1,0.1,(K,D))
        Beta  = np.append(Beta,random_betas,axis=1)
    if(ZEROS == init_flag):
        Alpha = np.zeros((D,M+1))
        Beta  = np.zeros((K,D+1))
    #print Alpha,Beta
    #Alpha = np.array([[1,1,2,-3,0,1,-3],[1,3,1,2,1,0,2],[1,2,2,2,2,2,1],[1,1,0,2,1,-2,2]],dtype=float)
    #Beta  = np.array([[1,1,2,-2,1],[1,1,-1,1,2],[1,3,1,-1,1]],dtype=float)
    return Alpha,Beta

#Turn scalar v into a one hot encoding vector V, zero,indexed
def one_hot_encoding(v,K):
    V = np.zeros((K,1))
    V[int(v)] = 1.0
    return V

def LinearForward(a,w):
    b = np.dot(w,a)
    return b

def LinearBackward(Z,W,Gb):
    Gw = np.dot(Gb,Z.T)
    Gz = np.dot(W.T,Gb)
    return Gw,Gz

def SigmoidForward(A,Alpha):
    b = sigmoid(A)
    return b

def SigmoidBackward(A,Z,Gz):
    b = np.multiply(Gz,np.multiply(Z,(1-Z)))
    return b

def SoftmaxForward(B,Beta):
    exp_B = np.exp(B)
    exp_sum = np.sum(exp_B,axis=0)
    b = np.divide(exp_B,exp_sum)
    return b

def SoftmaxBackward(B,Y_hat,Gy_hat):
    b = np.dot(Gy_hat.T,(np.diag(Y_hat) - np.outer(Y_hat,Y_hat.T)))
    return b

def CrossEntropyForward(Y,Y_hat):
    b = -np.dot(Y.T,np.log(Y_hat))
    return b

def CrossEntropyBackward(Y,Y_hat,J,Gj):
    b = np.negative(np.divide(Y,Y_hat))
    return b

def NNForward(X,Y,Alpha,Beta):
    X = np.reshape(X,(X.shape[0],1))
    A = LinearForward(X,Alpha)
    Z = SigmoidForward(A,Alpha)
    Z = np.insert(Z,0,1.0)
    Z = np.reshape(Z,(Z.shape[0],1))
    B = LinearForward(Z,Beta)
    Y_hat = SoftmaxForward(B,Beta)
    Y = np.reshape(Y,(Y.shape[0],1))
    J = CrossEntropyForward(Y,Y_hat)
    o = intermediates(X,A,Z,B,Y_hat,J)
    return o

def NNBackward(X,Y,Alpha,Beta,O):
    Gj = 1
    J = O.J
    Y_hat = O.y_hat
    B = O.b
    B = np.reshape(B,(B.shape[0],1))
    Z = O.z
    A = O.a
    X = np.reshape(X,(X.shape[0],1))
    Y = np.reshape(Y,(Y.shape[0],1))
    Gb = Y_hat - Y
    Gbeta,Gz = LinearBackward(Z,Beta,Gb)
    Z = Z[1:]
    Gz = Gz[1:]
    Ga = SigmoidBackward(A,Z,Gz)
    Galpha,Gx = LinearBackward(X,Alpha,Ga)
    return Galpha,Gbeta

def predict(data,Alpha,Beta,K):
    Y = data[0]
    X = data[1]
    
    N = X.shape[0]
    X =  np.append(np.ones((N,1)),X,axis=1)

    labels = []
    for x,y in zip(X,Y):
        one_hot_y = one_hot_encoding(y,K)
        obj_inter = NNForward(x,one_hot_y,Alpha,Beta)
        y_hat = obj_inter.y_hat
        label = np.argmax(y_hat)
        labels.append(int(label))
    return labels

def SGD(train_data,test_data):
    Y_train = train_data[0]
    X_train = train_data[1]
    Y_test = test_data[0]
    X_test = test_data[1]

    classes = set(Y_train)

    D = hidden_units
    K = len(classes)
    K = 10
    M = X_train.shape[1]
    N = X_train.shape[0]
    N_test = X_test.shape[0]
    Alpha,Beta = init_weights(D,K,M)
    
    X_train = np.append(np.ones((N,1)),X_train,axis=1)
    X_test = np.append(np.ones((N_test,1)),X_test,axis=1)
    
    train_out = open(train_out_file,"w+")
    test_out  = open(test_out_file, "w+")
    metrics_out = open(metrics_out_file,"w+")

    for e in range(epochs):
        J_total = 0.0
        for X,y in zip(X_train,Y_train):
            Y = one_hot_encoding(y,K)
            obj_inter = NNForward(X,Y,Alpha,Beta)
            Ga,Gb = NNBackward(X,Y,Alpha,Beta,obj_inter)
            Alpha = Alpha - eta*Ga
            Beta  = Beta  - eta*Gb
        for X,y in zip(X_train,Y_train): 
            Y = one_hot_encoding(y,K)
            obj_inter = NNForward(X,Y,Alpha,Beta)
            J_total += obj_inter.J 
        train_entr =  "epoch="+str(e+1)+" crossentropy(train): "+str(J_total/N) + "\n"
        metrics_out.write(train_entr)
        J_total = 0.0
        for X,y in zip(X_test,Y_test): 
            Y = one_hot_encoding(y,K)
            obj_inter = NNForward(X,Y,Alpha,Beta)
            J_total += obj_inter.J 
        test_entr = "epoch="+str(e+1)+" crossentropy(test): "+str(J_total/N_test) + "\n"
        metrics_out.write(test_entr)
    train_labels = predict(train_data,Alpha,Beta,K)
    correct = 0
    mistake = 0
    for (label,gold_label) in zip(train_labels,Y_train):
        train_out.write(str(label)+"\n")
        if(label == gold_label):
            correct += 1
        else:
            mistake += 1
    train_error =  "error(train): "+str(mistake/float(correct+mistake)) + "\n"
    metrics_out.write(train_error)
    test_labels = predict(test_data,Alpha,Beta,K)
    correct = 0
    mistake = 0
    for (label,gold_label) in zip(test_labels,Y_test):
        test_out.write(str(label)+"\n")
        if(label == gold_label):
            correct += 1
        else:
            mistake += 1
    test_error = "error(test): "+str(mistake/float(correct+mistake)) + "\n"
    metrics_out.write(test_error)
    return Alpha,Beta                     

def extract_data(ip_file):
    data = ip_file.readlines()
    raw_data = []
    for line in data:
        l_list = line.split(",")
        l_list[-1] = l_list[-1].strip('\n')
        raw_data.append(l_list)

    YnX = np.array(raw_data,dtype=float)
    Y = np.array(YnX[:,0])
    X = np.array(YnX[:,1:])
    return (Y,X)

#initialization schemes
RANDOM  = 1
ZEROS   = 2

train_in_file = sys.argv[1]
test_in_file = sys.argv[2]
train_out_file = sys.argv[3]
test_out_file = sys.argv[4]
metrics_out_file = sys.argv[5]
epochs = int(sys.argv[6])
hidden_units = int(sys.argv[7])
init_flag = int(sys.argv[8])
eta = float(sys.argv[9])

train_in = open(train_in_file,'r')
train_data = extract_data(train_in)
train_in.close()

test_in = open(test_in_file,'r')
test_data = extract_data(test_in)
test_in.close()

SGD(train_data,test_data)
