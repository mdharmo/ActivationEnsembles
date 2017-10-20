# Author: Mark Harmon
# Purpose:  This is my mean driver code for running a feed-forward network
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')
import matplotlib
matplotlib.use('pdf')
from NNResidualZ import main
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

# Need to go through each data set.

dataset = '/home/mharmon/ZProject/Data/cifarfinal.p'


numavg = 5
storeacc = np.zeros(numavg)
storeloss = np.zeros(numavg)
bn = True
bndir = '/BN'
for a in range(1):
    storeacc = np.zeros(numavg)
    storeloss = np.zeros(numavg)
    for t in range(numavg):
        np.random.seed(t)
        actsave = '/Act_' + str(a)
        actchoice = a+1
        num_epochs = 1500

        pngcount=0
        choose = 'conv'


        step = 0.001
        zstep = 0.001
        save = '/home/mharmon/ZProject/ModelResults/cifar/ZAllCNN' + bndir + actsave

        figsave = save + '/Figures/'
        parname = save + '/Parameters'

        testloss,testcorrect = main(actchoice = a)


        storeacc[t] = testcorrect
        storeloss[t] = testloss

print('Final Averages:')
print('Accuracy:\t\t\t {:.6f} %'.format(np.mean(storeacc)))
print('Loss:\t\t\t {:.6f}'.format(np.mean(storeloss)))

# Here is where I make a table
savestring = figsave + 'TableFinal.png'
cell_text = [[np.mean(storeloss), np.mean(storeacc)]]
rows = ['Test']
cols = ['Loss', 'Accuracy']
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=cols, loc=0)
plt.savefig(savestring)
plt.close()

