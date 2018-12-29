#!/usr/bin/env python

from matplotlib import pyplot as plt

x_axis = [5,20,50,100,200]
train_list = [0.53504588,0.13269205,0.05503127,0.04719075,0.0461534]
test_list  = [0.71434463,0.54991656,0.49895083,0.42433794,0.43043188]

plt.plot(x_axis,train_list,'b-',marker='^',label='train cross-entropy')
plt.plot(x_axis,test_list ,'g-',marker='x',label='test cross-entropy')
plt.ylabel("Average Cross Entropy")
plt.xlabel("Hidden Units")
plt.axis([0,205,0,0.75])
plt.legend(loc = 'best', shadow = True, fontsize = 'x-large' )
plt.show()
