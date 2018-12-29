#!/usr/bin/env python3
import sys
from collections import defaultdict,Counter,OrderedDict

start_state = "START"
end_state   = "END"

train_ip_file = sys.argv[1]
i2w_file = sys.argv[2]
i2t_file = sys.argv[3]
hmmprior_file = sys.argv[4]
hmmemit_file = sys.argv[5]
hmmtrans_file = sys.argv[6]

priors = defaultdict(float)
transitions = {}
emissions   = {}
emissionsTotal = defaultdict(float)
transitionsTotal = defaultdict(float)

vocab = OrderedDict()
tag_dict = OrderedDict()
i2w = open(i2w_file,"r")
for (index,word) in enumerate(i2w):
    word = word.rstrip("\n")
    vocab[word] = index

i2t = open(i2t_file,"r")
for (index,tag) in enumerate(i2t):
    tag = tag.rstrip("\n")
    print (index,tag)
    tag_dict[tag] = index

#print vocab
#print tag_dict

train_ip = open(train_ip_file,"r")
for line in train_ip:
    line  = line.rstrip("\n")
    start = line.split(" ")[0]
    parts = start.split("_")
    tag   = parts[1]
    priors[tag] += 1

    tag_tokens = line.split(" ")
    prevtag = start_state
    for tag_token in tag_tokens:
        parts = tag_token.split("_")
        token = parts[0]
        tag   = parts[1]
        #print (token+" "+tag)
        #print (str(prevtag)+" "+ tag)
        token = vocab[token]
        tag = tag_dict[tag]
        #print (str(token)+" "+str(tag))
        #print (str(prevtag)+" "+ str(tag))
        #print ""
        if tag not in emissions:
            emissions[tag] = defaultdict(float)

        emissions[tag][token] += 1
        emissionsTotal[tag] += 1

        if prevtag == "START":
            prevtag = tag
            continue
        if prevtag not in transitions:
            transitions[prevtag] = defaultdict(float)
        transitions[prevtag][tag] += 1
        transitionsTotal[prevtag] += 1
        
        prevtag = tag

hmmprior = open(hmmprior_file,"w+")
#calculate the probabilities 
priorsTotal = sum(priors.values())
#print priors
print tag_dict
for key in tag_dict:
    priors[key] = (priors[key]+1)/(priorsTotal+len(tag_dict))
    hmmprior.write(str(priors[key])+"\n")
    print (str(priors[key])+"\n")

hmmtrans = open(hmmtrans_file,"w+")
print "transitions"
for prevtag in transitions:
    write_str = ""
    for tag in transitions:
        transitions[prevtag][tag] += 1
        transitions[prevtag][tag] = transitions[prevtag][tag]/(transitionsTotal[prevtag]+len(tag_dict))
        print (str(prevtag)+" "+str(tag)+" "+str(transitions[prevtag][tag]))
        write_str += str(transitions[prevtag][tag])+" "
    write_str  = write_str.rstrip(" ")
    write_str += "\n"
    hmmtrans.write(write_str)

hmmemit = open(hmmemit_file,"w+")
#print "emissions"
for tag in emissions:
    write_str = ""
    for token in vocab.values():
        emissions[tag][token] += 1
        emissions[tag][token] = emissions[tag][token]/(emissionsTotal[tag]+len(vocab)) 
        #print (str(tag)+" "+str(token)+" "+str(emissions[tag][token]))
        write_str += str(emissions[tag][token])+" "
    write_str = write_str.rstrip(" ")
    write_str += "\n"
    hmmemit.write(write_str)
