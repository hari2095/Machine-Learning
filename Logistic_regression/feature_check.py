#!/usr/bin/env python
import sys

def strip_newline(arr):
    for i in range(len(arr)):
        if '\n' in arr[i]:
            arr[i] = arr[i].rstrip("\n")
    return arr

file1 = sys.argv[1]
file2 = sys.argv[2]

text1 = open(file1,'r').read()
text2 = open(file2,'r').read()

tokens1 = text1.split("\t")
#tokens1 = tokens1.split("\t")
tokens2 = text2.split("\t")
#tokens2 = tokens2.split("\t")

tokens1 = strip_newline(tokens1)
tokens2 = strip_newline(tokens2)

print len(tokens1)
print len(tokens2)

diff_list = list(set(tokens1) - set(tokens2))
diff_list1 = [elem for elem in diff_list if '\n' not in elem]

print diff_list
