# Author: Mark Harmon
# Purpose:  This is my mean driver code for running a feed-forward network
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu0')
import matplotlib
matplotlib.use('pdf')
from NNAutoZ import main
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
# Need to go through each data set.

set1 = '/home/mharmon/ZProject/Data/stlfinaltrain/'
set2 = '/home/mharmon/ZProject/Data/stlfinalval/'
set3 = '/home/mharmon/ZProject/Data/stlfinaltest/'

numavg = 1
storeacc = np.zeros(numavg)
storeloss = np.zeros(numavg)
save = '/home/mharmon/ZProject/ModelResults/stl/Z'
num_epochs =75
parname = save + '/Parameters'
figsave = save+'/Figures/'
storesave = save
actchoice = 2
t=numavg
np.random.seed(t)

testloss = main(set1,set2,set3,num_epochs,parname,storesave,actchoice)




