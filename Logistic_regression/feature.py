#!/usr/bin/env python3
import sys
import math
import csv
import re
from collections import Counter

sample = "0	stallone attempts to 'act' in this cop drama . the film is set in a neighbourhood pratically built by kietal , who's nephew ( played by michael rappaport ) is involved in a car crash and killing of two black youths . keital dosen't really want to get involved in anything , gets rid of rappaport , and stallone and de niro try to work out what the hell is going on . this film should be brilliant . it sounds like a great plot , the actors are first grade , and the supporting cast is good aswell , and stallone is attempting to deliver a good performance . however , it can't hold up . although the acting is fantastic ( even stallone isn't bad ) the directing and story is dull and long winded some scenes go on for too long , with nothing really happening in them . in fact , the only scenes that do work are action scenes , which i suspect stallone was trying to avoid . in this film , serious means dull . the dialogue is warbling , and basically just repeats the same points over and over , no matter who is delivering them . the plot , which has potential , is wasted , again just being cliched after a while . in fact , the only thing that does keep the film going is kietal and de niro , both delivering their usual good performances . however , stallone , although not given much to say , gives a good performance . however , it's not all that bad . as said above , the action scenes are well done . theres also a very good ending , which uses the cinemas sound system well . in fact , the last 10 minutes of this 2 hour film are one of the best endings of 1997 . if only the rest of the film was as good as the ending . cop land , then , turns out not to be a power house film , but a rather dull , and not every exciting film . hugely disappointing , and i can't really recommend it ."

sample = "1	clint eastwood , in his ripe old age , is cashing one talent in for another . midnight in the garden of good and evil is an eastwood-directed film clint isn't even in , and it's damn good . adapted from a best-selling john berendt novel based on true events , this movie is set in the bizarre georgia town of savannah , where people walk invisible dogs and attach horseflies to their head . and that's just the mayor . as director and producer , eastwood contributes a self- indulgent but very competent 150 minutes , neatly balancing drama , suspense and humor . like all great movies , midnight in the garden has a lot of funny moments that spring from the characters themselves , and not from some contrived , juvenile intrusion . the acting , from big-time stars john cusack and kevin spacey , is as good as you'd expect , but it's the supporting stable that gives the movie its offbeat charm . cusack plays a free-lance reporter sent to savannah to write a fluff story for town & country magazine about one of eccentric millionaire spacey's parties . lots of food , beverages and gunplay . yes , an employee of spacey's has a huge argument in front of cusack and later turns up dead . it seems like a simple matter of self- defense -- the guy threatened spacey , shot at him , missed , then spacey took him out -- but being a movie , there's much more to it than that , and cusack decides to stay in town and write a book about the murder , a book which will eventually become a movie he will star in . the weird circle of entertainment . this is where the stable of supporting characters comes in . there's the requisite sexy woman ( alison  nepotist's daughter  eastwood ) , the strange piano player , the voodoo woman and the transvestite . miss chablis deserves a paragraph of his/her own , as the stealer of every scene she appears in -- what would this movie be without the castilian scene and her testimony ? chablis , playing herself , is five times funnier than reigning drag queen rupaul , but never seems exploited as the movie's comic relief or out of place in a basically serious movie . there are also a fair share of courtroom scenes , which these days almost never seem welcome in a movie , but even here clint manages to keep the movie fascinating . certainly the characterizations of the judge and spacey's lawyer both help immensely , as does having the fly-guy as the jury's foreman . it's here we realize the case boils down to an indictment of spacey's homosexuality . yes , spacey had a sexual relationship with the dead man -- before he shot him , of course . midnight in the garden of good and evil isn't a classic , but it is one of the better celebrity-directed , true-story best-seller adaptations out there . likewise , no one from this movie will be getting any oscar nominations , but the performances are all great . in fact , i'd almost recommend that the academy add a best supporting transvestite category . serving america for more than 1/25th of "

def extract_features(ip_file,op_file,dict_file,model):
    vocab_dict = {}
    dict_text = dict_file.readlines()
    #print model
    for line in dict_text:
        elems = line.split(" ")
        word = elems[0]
        index = elems[1]
        index = index.rstrip("\n")
        vocab_dict[word] = index
    text = csv.reader(ip_file,delimiter = '\t')
    op_text = ""
    keys = vocab_dict.keys()
    for row in text:
        label = row[0]
        words = row[1]
        #words = sample
        words_counter = Counter(words.split(" "))
        if 1 == int(model):
            words = set(words.split(" "))
        elif 2 == int(model):
            words = set(k for k,v in words_counter.iteritems() if v < 4)
        op_text += label + "\t"
        for word in keys:
            if word in words:
                op_text += str(vocab_dict[word]) + ":1\t"
        op_text = op_text.rstrip("\t")
        op_text += "\n"
    op_file.write(op_text)

train_in_file_name = sys.argv[1]
validation_in_file_name = sys.argv[2]
test_in_file_name = sys.argv[3]
dict_in_file_name = sys.argv[4]
fmt_train_out_file_name = sys.argv[5] 
fmt_validation_out_file_name = sys.argv[6]
fmt_test_out_file_name =sys.argv[7]
feature_flag = sys.argv[8]

train_in = open(train_in_file_name,'r')
dict_in  = open(dict_in_file_name,'r')
train_out = open(fmt_train_out_file_name,'w+')
extract_features(train_in,train_out,dict_in,feature_flag)
train_in.close()
dict_in.close()
train_out.close()


valid_in = open(validation_in_file_name,'r')
valid_out = open(fmt_validation_out_file_name,'w+')
dict_in  = open(dict_in_file_name,'r')
extract_features(valid_in,valid_out,dict_in,feature_flag)
valid_in.close()
dict_in.close()
valid_out.close()


test_in  = open(test_in_file_name,'r')
test_out  = open(fmt_test_out_file_name,'w+')
dict_in  = open(dict_in_file_name,'r')
extract_features(test_in,test_out,dict_in,feature_flag)
test_in.close()
dict_in.close()
test_out.close()


