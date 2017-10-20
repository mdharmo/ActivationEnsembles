# Author: Mark Harmon
# Purpose:  This is my main driver code for running a feed-forward network for mnist set.

import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu2')
import matplotlib
matplotlib.use('pdf')
from NNDeconvReg import main
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

numavg = 1


for t in range(numavg):

    num_epochs = 350

    pngcount=0
    choose = 'conv'



    save = '/home/mharmon/ZProject/ModelResults/stl/Deconv/'

    figsave = save + '/Figures/'
    parname = save + '/Parameters'

    testloss,valloss,bestloss = main(num_epochs,parname)

