#!/usr/bin/env python2

import csv
import sys
import math
from collections import defaultdict,Counter

decision_label_attr = ""

def entropy(labels):
    total = 0
    label_freqs = []
    for key,value in labels.iteritems():
        #print "key,value",key,value
        total += value
        label_freqs.append(value)

    #print "label_freqs: ",label_freqs
    label_probs = [float(freq)/total for freq in label_freqs]
    #print "label_probs: ",label_probs
    ent = 0
    for prob in label_probs:
        if (prob > 0):
            ent += prob*math.log(prob,2)
    return ent*(-1)

"""
The inherent assumption is that list lengths are the same.
Partition Y on a value 'x' in X
X=[0,1,0,0,1,1,1]
Y=[y,y,n,n,n,n,n] 
x = 1
partition_list(Y,X,x) returns [y,n,n,n]
List order is preserved here.
"""
def partition_list(Y,X,x):
    indices = [idx for idx,x1 in enumerate(X) if x==x1]
    Y_x = [Y[idx] for idx in indices] 

    return Y_x

def create_labels(X):
    labels = {}

    for element in X:
        if element in labels:
            labels[element] += 1
        else:
            labels[element] = 1    
    
    return labels

"""
Given two lists like:
X=[0,1,0,0,1,1,1]
Y=[y,y,n,n,n,n,n] 

Where Y represents the column of decisions taken
and X represents the attribute values,
compute I(Y,X) 
"""
def mutual_information(Y,X):
    #I(Y;X) = H(Y) - H(Y|X)
    #I(Y;X) = H(Y) - sum( P(Y|X=i)*H(Y|X=i) ) 
    #where "i" takes various values of the attributes (discrete only)

    labels_y = create_labels(Y)
    
    #print "Y: ",Y
    #print "X: ",X
    #calculate the first term H(Y)
    H_y = entropy(labels_y)
    #print "H(Y): "+str(H_y)
    cond_prob = 0

    #list of possible values for the attribute, e.g. 'yes' or 'no'
    x_vals = list(set(X))

    x_vals_probs = [float(X.count(x))/len(X) for x in x_vals]    

    H_Y_xi = 0
    H_Y_x  = 0 
    for i in range(len(x_vals)):
        Y_part_x = partition_list(Y,X,x_vals[i])
        #print "Y split on ",x_vals[i]," ",Y_part_x
        labels_Y_x = create_labels(Y_part_x)

        H_Y_xi = entropy(labels_Y_x) 
        #print str(x_vals_probs[i])+ "*"+str(H_Y_xi)
        H_Y_x  += x_vals_probs[i]*H_Y_xi 
    
    #print "H(Y|X): "+str(H_Y_x)

    info_gain = H_y - H_Y_x
    return info_gain

def make_leaf(node,results):
    #We don't care about the attributes for a majority vote classifier.
    node.split_attr = None
    
    #print node.data.set_data(Counter(results).most_common(1)[0][0])
    #Here the data is only a single decision label string, e.g. 'play' or 'not play'
    node.label = (Counter(results).most_common(1)[0][0])
    
    #It's a leaf node, hence no children are present.
    #This is the test for identification of leaf nodes
    node.left  = None
    node.right = None

    return node

def train (data, attributes, decision_label_name,max_depth,curr_depth,uniques):
    
    #print data
    node = decisionTree(curr_depth)
    """
    reached max_depth,implement majority vote classifier here
    and make node a leaf node
    """
    results = data[decision_label_name]
    max_depth = int(max_depth)
    #print curr_depth
    #print max_depth

    #print curr_depth - max_depth
    if (curr_depth >= max_depth):
        #print "reached depth"
        node = make_leaf(node,results)
        return node 
    #print "echeked for max depth, fine"
    """
    If all the labels are either positive or negative, the attributes
    don't matter at this point, create node with the label value and return
    """
    #if root.left:
        #print root.left.data.get_data()
    #if root.right:
        #print root.right.data.get_data()
    #print  root.data.get_data()
    #print results
    #print Counter(results).most_common()
    if len(Counter(results).most_common()) == 1:
        #print "all positive/negative" 
        node = make_leaf(node,results)
        return node

    """
    If there are no more attributes to split on, return the majority label
    """    
    
    #print attributes
    #print "Checking if no more attributes"
    if not attributes:
        #print "no more attributes to split on"
        node = make_leaf(node,results)
        return node
    
    #if flow reaches here, proceed to partition on an attribute x
    information_gain = {}

    max_gain = 0
    attr_to_split = ""
    for attr in attributes:
        information_gain[attr] = mutual_information(results,data[attr])
        #print "for " +attr 
        #print information_gain[attr]
        if (information_gain[attr] > max_gain):
            max_gain = information_gain[attr]
            attr_to_split = attr

    if (max_gain <= 0):
        #print "No benefit on splitting any further"
        node = make_leaf(node,results)
        return node

    node.split_attr = attr_to_split
    list_to_split_on = data[attr_to_split]
    unique_vals = list(set(list_to_split_on))
    #print list_to_split_on
    data_table = defaultdict(list)

    #NOTE!! ATTRIBUTE TO SPLIT ON BEING REMOVED HERE!!
    rem_attributes = list(attributes)
    rem_attributes.remove(attr_to_split)
    #print attributes,attr_to_split
    #if attr in attributes:
    #    attributes.remove(attr_to_split)

    iteration = 0
    
    #construct data tables for children nodes
    for val_to_split_on in unique_vals:
        for attr in rem_attributes:
            part_list = partition_list(data[attr],list_to_split_on,val_to_split_on)
            data_table[attr] = part_list
        result_part_list = partition_list(data[decision_label_name],list_to_split_on,val_to_split_on)
        data_table[decision_label_name] = result_part_list
        #print data_table[decision_label_name]
        #Node = decisionTree(data_table,new_attributes,root.depth+1)
        
        print_depth = curr_depth+1
        depth_marker = '|'*print_depth
        #list_tuple_counts = list(Counter(result_part_list).elements())
        #for element in list_tuple_counts:
        #print list_tuple_counts
        #list_tuple_counts += Counter()
        y_splits = ""
        for element in uniques[decision_label_name]:
            y_splits += str(result_part_list.count(element))+" "+str(element)+" /"
       
        y_splits = y_splits.rstrip(" /")
        to_print = depth_marker+" "+attr_to_split+" = "+val_to_split_on+": ["+y_splits+"]" 
        print to_print 
        if(iteration == 0):
            #print "going left"
            #root.left = train(root.left,depth,decision_label_name) 
            node.leftedge = val_to_split_on
            node.left = train(data_table, rem_attributes, decision_label_name,max_depth,curr_depth+1,uniques)
            
        #if (root.left.left != None):
            #root.left = make_leaf(node,root.left,root.left.data.get_data()[decision_label_name])
        if(iteration == 1):
            #print "going right"
            #root.right = train(root.right,depth,decision_label_name)
            node.rightedge = val_to_split_on
            node.right = train(data_table, rem_attributes, decision_label_name,max_depth,curr_depth+1,uniques)
        iteration += 1
    
    return node
    #print "returning from right sub tree"       

def predict_helper(tree,decision_attr,data,uniques):
    #base case, check if we have reached one of the leaf nodes
    if(None == tree.left and None == tree.right):
        if(tree.label):
            label = tree.label
            return label
        else:
            #for debugging
            #If you see the following print statement there's a bug in the code
            print "UNinited leaf"
    
    attr = tree.split_attr
    attr_val = data[attr]
    if (tree.left != None and tree.leftedge == attr_val):
        label = predict_helper(tree.left,decision_attr,data,uniques)
        return label
    if(tree.right != None and tree.rightedge == attr_val):
        label = predict_helper(tree.right,decision_attr,data,uniques)
        return label
   
    #if the flow reaches here, it means that the attribute value
    # does not have a branch at this level, we use the other value from 
    #uniques
    
    alt_attr_val = ""
    for element in uniques:
        if attr_val != element:
            alt_attr_val = element

    if (tree.left != None and tree.leftedge == alt_attr_val):
        label = predict_helper(tree.left,decision_attr,data,uniques)
        return label
    if(tree.right != None and tree.rightedge == alt_attr_val):
        label = predict_helper(tree.right,decision_attr,data,uniques)
        return label

    
    
def predict(tree,input_file_name,output_file_name,metrics_file_name,ip_file_type,uniques):

    #print "prediction stage" 
    data_table = defaultdict(list) 
    ip_file = open(input_file_name,"r")
    op_file = open(output_file_name,"w")
    metrics_file = open(metrics_file_name,"a")

    attributes = []
    csv_data = csv.DictReader(ip_file)
    decision_label_attr = csv_data.fieldnames[len(csv_data.fieldnames)-1] 
    read_attr_names = True
    #i = 20
    correct = 0
    mistake = 0
    for row in csv_data:
        label = predict_helper(tree,decision_label_attr,row,uniques)
        op_file.write(label+"\n")
        #if i > 0:
            #print row
            #print label
            #i -= 1
        if (label == row[decision_label_attr]):
            correct += 1
        else:
            mistake += 1
    ip_file.close()
    op_file.close()
    #print correct
    #print mistake
    
    error_rate = float(mistake)/(mistake+correct)
    metrics_file.write("error("+ip_file_type+"): "+str(error_rate)+"\n")
    metrics_file.close()
    """   
    for attr_name,attr_value in row.iteritems():
        if (read_attr_names):
            if(attr_name != decision_label_attr):
                attributes.append(attr_name)
                #while reading the values for the first row,
                #sniff out the names for the attributes too
        data_table[attr_name].append(attr_value)
    read_attr_names = False
    """

    #attr_to_pass = list(attributes)
    
    #for row in csv_data:
        #label = predict_helper(tree,decision_attr,csv_data[row])
        #print label     
    #curr_depth = 0
     
class decisionTree:
    def __init__(self,depth):
        self.left  = None
        self.right = None
        self.leftedge = None
        self.rightedge = None
        self.depth = depth
        self.label = None
        self.split_attr = None
args = sys.argv

train_in_name    = args[1]
test_in_name     = args[2]
max_depth        = args[3]
train_out_name   = args[4]
test_out_name    = args[5]
metrics_out_name = args[6]

attributes = []

uniques = {}
data_table = defaultdict(list)
rootNode = decisionTree(0)

with open(train_in_name,'r') as train_in:
    csv_data = csv.DictReader(train_in)
    read_attr_names = True
    decision_label_attr = csv_data.fieldnames[len(csv_data.fieldnames)-1] 
    #print decision_label_attr
    #print csv_data.fieldnames
    #print csv_data
    i=0
    for row in csv_data:
        for attr_name,attr_value in row.iteritems():
            if (read_attr_names):
                if(attr_name != decision_label_attr):
                    attributes.append(attr_name)
                #while reading the values for the first row,
                #sniff out the names for the attributes too
            data_table[attr_name].append(attr_value)
        read_attr_names = False
    
    for attr in attributes:
        uniques[attr] = list(set(data_table[attr]))
    uniques[decision_label_attr] = list(set(data_table[decision_label_attr])) 
    #print uniques
    curr_depth = 0
    y_splits = ""
    for element in uniques[decision_label_attr]:
        y_splits += str(data_table[decision_label_attr].count(element))+" "+str(element)+" /"
       
    y_splits = y_splits.rstrip("/")
    print "["+y_splits+"]"
    rootNode = train(data_table, attributes, decision_label_attr,max_depth,curr_depth,uniques)

"""
curr_depth = 0
for attr in attributes:
    rootNode = decisionTree(0)
    print curr_depth
    rootNode = train(data_table, attributes, decision_label_attr,max_depth,curr_depth,uniques)   
    
    input_file_name = train_in_name
    output_file_name = train_out_name
    metrics_file_name = "pol"+str(curr_depth)+"metrics.txt"

    
    ip_file_type = "train"
    predict(rootNode,input_file_name,output_file_name,metrics_file_name,ip_file_type,uniques)
    
    
    input_file_name = test_in_name
    output_file_name = test_out_name

    ip_file_type = "test"
    predict(rootNode,input_file_name,output_file_name,metrics_file_name,ip_file_type,uniques)
    curr_depth += 1
    
"""
#print uniques

input_file_name = train_in_name
output_file_name = train_out_name
metrics_file_name = metrics_out_name
   
ip_file_type = "train"
predict(rootNode,input_file_name,output_file_name,metrics_file_name,ip_file_type,uniques)

input_file_name = test_in_name
output_file_name = test_out_name
metrics_file_name = metrics_out_name
   
ip_file_type = "test"
predict(rootNode,input_file_name,output_file_name,metrics_file_name,ip_file_type,uniques)


    #print attributes
    #rootNode = decisionTree(data_table,attributes,0)
    
    #train (rootNode,max_depth,decision_label_attr)
    #for attr in attributes:
        #print attr,":",data_table[attr]
