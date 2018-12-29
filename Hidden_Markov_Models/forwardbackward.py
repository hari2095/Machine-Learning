#!/usr/bin/env python
import sys
from collections import defaultdict,OrderedDict,Counter
import numpy as np

START_STATE = "START"
END_STATE   = "END"
terminal_states = [START_STATE,END_STATE]
mistake = 0
correct = 0


ip_file = sys.argv[1]
i2w_file = sys.argv[2]
i2t_file = sys.argv[3]
hmmprior_file = sys.argv[4]
hmmemit_file = sys.argv[5]
hmmtrans_file = sys.argv[6]
predicted_file = sys.argv[7]
metric_file = sys.argv[8]

def predict(Alpha,Beta,line):
    l = line.split(" ")
    sent = ""
    correct = 0
    for t in range(0,len(l)):
        parts = l[t].split("_")
        token = parts[0]
        gold_tag = parts[1]
        #print Alpha[t]
        #Alpha[t] = np.insert(Alpha[t],4,0.0)
        #print Alpha[t]
        #print t
        if (8 == Alpha[t].shape[0]):
            #try:
            Alpha[t] = np.insert(Alpha[t],4,0.0)
            #print Alpha[t]
            #except ValueError as v:
            #    print Alpha[t]
        predicted_tag = tag_reverse_lookup[np.argmax(Alpha[t]*Beta[t])]
        if predicted_tag == gold_tag:
            correct += 1
        sent += token + "_" + predicted_tag + " "
    sent = sent.rstrip(" ")
    sent += "\n"
    return correct,sent

    
def normalize(vect):
    vect = vect/np.sum(vect)
    return vect

def npdict2np(npdict):
    l = []
    for row in npdict:
        l.append(npdict[row])
    l = np.array(l)
    return l

def od2np(od):
    l = []
    for row in od:
        l.append(map(float,od[row].values()))
    l = np.array(l)
    return l

def element_wise_pdt(vec, mat):
    mat_pdt = []
    for r1,row in zip(vec,mat):
        row = np.array([float(r1)*float(e) for e in row])
        mat_pdt.append(row) 
    mat_pdt = np.array(mat_pdt)
    return np.asarray(mat_pdt)

def transpose(matrix):
    a = [matrix[row].values() for row in matrix]
    aT = np.array(a).T
    return aT

def extract_column(matrix,key):
    key = vocab[key]
    column = []
    for row in matrix:
        col_val =  matrix[row][key]
        column.append(float(col_val))
    return column

#Load pre-computed data
vocab = OrderedDict()
tag_dict = OrderedDict()
i2w = open(i2w_file,"r")
for (index,word) in enumerate(i2w):
    word = word.rstrip("\n")
    vocab[word] = index

tag_reverse_lookup = []
i2t = open(i2t_file,"r")
for (index,tag) in enumerate(i2t):
    tag = tag.rstrip("\n")
    tag_dict[tag] = index
    tag_reverse_lookup.append(tag)


states = []
states.append(START_STATE)
for state in tag_dict:
    states.append(state)
states.append(END_STATE)

pi = {}
hmmprior_f = open(hmmprior_file,"r")
for (index,prior) in enumerate(hmmprior_f):
    prior = prior.rstrip("\n")
    pi[index] = prior

A = OrderedDict()
B = OrderedDict()

hmmtrans_f = open(hmmtrans_file,"r")
for (row,line) in enumerate(hmmtrans_f):
    line = line.rstrip("\n")
    trans = line.split(" ")
    A[row] = {}
    for (col,t) in enumerate(trans):
        A[row][col] = t

aT = transpose(A)
hmmemit_f = open(hmmemit_file,"r")
for (row,line) in enumerate(hmmemit_f):
    line = line.rstrip("\n")
    emit = line.split(" ")
    B[row] = {}
    for (col,token) in enumerate(emit):
        B[row][col] = token

A = od2np(A)
ip = open(ip_file,"r")
lines = ip.readlines()
loglikehoods = []
total_examples = 0

predicted = open(predicted_file,"w+")

correct = 0
mistake = 0
#for each sequence of tokens, run the forward backward algorithm
r = 0
for line in lines:
    line = line.rstrip("\n")
    Alpha = {}
    Beta  = {}    

    Alpha[1] = []
    X1 = line.split(" ")[0]
    parts = X1.split("_")
    token = parts[0]
    tag = parts[1]

    X1_column = extract_column(B,token)
    for i in range(len(X1_column)):
        Alpha[1].append(float(X1_column[i])*float(pi[i]))
    
    Alpha[1] = np.array(Alpha[1])

    l = line.split(" ")
    if len(l) > 1:
        Alpha[1] = normalize(Alpha[1])
    for i in range(1,len(l)):
        Alpha[i] = np.insert(Alpha[i],4,0.0)
        Xi = l[i]
        parts  = Xi.split("_")
        token = parts[0]
        tag   = parts[1]
        Xi_column = extract_column(B,token)
        Xi_np = np.array(Xi_column)
        bXiaT = element_wise_pdt(Xi_np,aT)
        bX = np.array(bXiaT)
        #print bX.shape
        #print Alpha[i].shape
        #print Alpha
        Alpha[i+1] = np.dot(bX,Alpha[i])
        if  len(l)> 1 and i < (len(l)-1):
            Alpha[i+1] = normalize(Alpha[i+1])
        
    Beta[len(l)] = np.array([1.0 for e in tag_dict])        
    Beta[len(l)] = normalize(Beta[len(l)])
    for i in range(len(l),1,-1):
        Xi = l[i-1]
        parts  = Xi.split("_")
        token = parts[0]
        tag   = parts[1]
        Xi_column = np.array(extract_column(B,token))
        #print Xi_column.shape
        #print Beta[i].shape
        Xi_column = np.insert(Xi_column,4,0.0)
        b_mat = np.multiply(Xi_column,Beta[i])
        b_mat = np.delete(b_mat,4) 
        Beta[i-1] = np.dot(A,b_mat)
        Beta[i-1] = normalize(Beta[i-1])
        
    Alpha = npdict2np(Alpha)
    Beta  = npdict2np(Beta)
    r += 1
    #print r
    if len(l) == 1:
        continue
    c,prediction = predict(Alpha,Beta,line)
    predicted.write(prediction)
    correct += c
    mistake += len(l)
    
    loglikehoods.append(np.log(np.sum(Alpha[-1])))
    
predicted.close()

loglikehoods = np.array(loglikehoods)
avg_ll = np.average(loglikehoods)
print avg_ll

mistake = mistake - correct
print (correct/float(correct+mistake))
met = open(metric_file,"w+")
met.write("Average Log - Likelihood : "+str(avg_ll)+"\n")
met.write("Accuracy: "+str(correct/float(correct+mistake))+"\n")
met.close()
