# Author: Mark Harmon
# Purpose:  This is my main driver code for running a feed-forward network for mnist set.

import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu2')
import matplotlib
matplotlib.use('pdf')
from NNFeedReg import main
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
# Need to go through each data set.
from theano.tensor.shared_randomstreams import RandomStreams

srng = RandomStreams(seed=234)

dataset = '/home/mharmon/ZProject/Data/mnist.p'


avglen = 20

storeloss = np.zeros(avglen)
storeacc = np.zeros(avglen)
for k in range(avglen):
    for b in range(1):
        seed = k*10+1
        bn = True
        bndir = '/BN'


        num_epochs = 2000

        pngcount=0
        choose = 'conv'


        step = 1.

        save = '/home/mharmon/ZProject/ModelResults/mnist/FeedFinalAda' + bndir

        figsave = save + '/Figures/'
        parname = save + '/Parameters'

        testloss,testcorrect,testpred,valloss,valcorrect,valpred,bestloss,bestperc = main(dataset,num_epochs,parname,step,bn,seed)

        # Make histograms
        # First, lets plot the validation error and test errors


        x1 = [x for x in range(0,len(valcorrect))]
        x2 = [x for x in range(0,len(testcorrect))]


        if not os.path.exists(figsave):
            os.makedirs(figsave)
        else:
            shutil.rmtree(figsave)
            os.makedirs(figsave)
        '''
        name = figsave + 'Val_Error.png'
        plt.figure(pngcount)
        plt.plot(x1,valcorrect,linewidth=2.0)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('Validation Error Plot')
        plt.savefig(name)
        plt.close()
        pngcount+=1


        x3 = [x for x in range(0,len(valloss))]
        name = figsave + 'Cost.png'
        plt.figure(pngcount)
        plt.plot(x3,valloss,linewidth=2.0)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Cost Plot (Validation Data)')
        plt.savefig(name)
        plt.close()
        pngcount += 1

        # Get total Histogram with all roles:
        bins = np.linspace(0,1,100)
        testpred = np.array(testpred)
        myargmax = np.argmax(testpred,axis=1)

        plottest = np.zeros(myargmax.shape)
        for i in range(len(myargmax)):
            plottest[i] = testpred[i,myargmax[i]]

        plt.figure(pngcount)
        titlestring = 'Test Probability Histogram (Highest Probability)'
        savestring = figsave + 'HistogramAllClasses.png'
        plt.hist(plottest,bins)
        plt.xlabel('Values (Bins)')
        plt.ylabel('Frequency')
        plt.savefig(savestring)
        plt.close()
        pngcount += 1
        '''

        # Here is where I make a table
        savestring = figsave + 'Table.png'
        cell_text = [[str(testloss[0]),str(testcorrect[0])],[str(bestloss),str(bestperc)]]
        rows = ['Test','Validation']
        cols = ['Loss','Accuracy']
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              colLabels=cols,loc=0)
        plt.savefig(savestring)
        plt.close()
        pngcount+=1

        storeloss[k] = testloss[0]
        storeacc[k] = testcorrect[0]


print('Final Averages:')
print('Accuracy:\t\t\t {:.6f} %'.format(np.mean(storeacc)))
print('Loss:\t\t\t {:.6f}'.format(np.mean(storeloss)))
storesave = figsave + 'StorageSave.npy'
np.save(storesave, storeacc)
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

savestring = figsave + 'AccArray.npz'
np.save(savestring,storeacc)
