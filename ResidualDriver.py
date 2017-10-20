# Author: Mark Harmon
# Purpose:  This is my mean driver code for running a feed-forward network
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu2')
import matplotlib
matplotlib.use('pdf')
from ResidualNetwork import main
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

# Need to go through each data set.

dataset = '/home/mharmon/ZProject/Data/cifarfinal.p'


numavg = 5
storeacc = np.zeros(numavg)
storeloss = np.zeros(numavg)
storeacc = np.zeros(numavg)
storeloss = np.zeros(numavg)
figsave = '/home/mharmon/ZProject/ModelResults/cifar/Reg/'
for t in range(numavg):
    np.random.seed(t)

    testloss,testcorrect = main()


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

