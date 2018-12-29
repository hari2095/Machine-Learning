#!/usr/bin/env python2
import sys
import csv
import math

def calc_entropy(labels):
    total = 0
    label_freqs = []
    for key,value in labels.iteritems():
        total += value
        label_freqs.append(value)

    label_probs = [float(freq)/total for freq in label_freqs]
    entropy = 0
    for prob in label_probs:
        if prob > 0:
            entropy += prob*math.log(prob,2)
    return entropy*(-1)

def calc_error(labels):
    total = 0
    max_val = 0
    for key,value in labels.iteritems():
        total += value
        if value > max_val:
            max_val = value
    error = float(total-max_val)/total
    return error

input_file_name  = sys.argv[1]
output_file_name = sys.argv[2]

entropy = 0
error = 0

labels = {}

with open(input_file_name,'r') as input_file:
    input_file.readline()
    #skip the first line
    csv_data = csv.reader(input_file)
    for row in csv_data:
        key = row[len(row) - 1]
        if key in labels:
            labels[key] += 1
        else:
            labels[key] = 1
    print labels
    entropy = calc_entropy(labels)
    error   = calc_error(labels)

    entropy_str = "entropy: "+str(entropy)+"\n"
    error_str = "error: "+str(error)+"\n"

    output_file = open(output_file_name,'w')
    output_file.write(entropy_str+error_str)

