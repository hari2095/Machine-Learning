#!/usr/bin/env python

from matplotlib import pyplot as plt

x_axis = [10,100,1000,10000]
train_list = [-7.8297,-6.5391,-5.4689,-4.8607]
test_list  = [-7.6314,-6.1174,-5.0022,-4.4226]

plt.plot(x_axis,train_list,'b-',marker='^',label='train log-likelihood')
plt.plot(x_axis,test_list ,'g-',marker='x',label='test log-likelihood')
plt.ylabel("Log-Likelihood")
plt.xlabel("Training Sequence Length")
plt.axis([-1000,11000,0,-8.5])
plt.legend(loc = 'best', shadow = True, fontsize = 'x-large' )
plt.show()
