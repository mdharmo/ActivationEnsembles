# Author: Mark Harmon
# Purpose:  This is my mean driver code for running a feed-forward network
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')
import matplotlib
matplotlib.use('pdf')
from NNConvZ2 import main
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

# Need to go through each data set.

dataset = '/home/mharmon/ZProject/Data/cifar.p'


numavg = 1
storeacc = np.zeros(numavg)
storeloss = np.zeros(numavg)
bn = True
bndir = '/BN'
for a in range(3):
    storeacc = np.zeros(numavg)
    storeloss = np.zeros(numavg)
    for t in range(numavg):

        actsave = '/Act_' + str(a)
        actchoice = a
        num_epochs = 1500

        pngcount=0
        choose = 'conv'


        step = 0.001
        zstep = 0.001
        save = '/home/mharmon/ZProject/ModelResults/cifar/ConvEnsembleAda' + bndir + actsave

        figsave = save + '/Figures/'
        parname = save + '/Parameters'

        testloss,testcorrect,testpred,valloss,valcorrect,valpred,zstore1,zstore3,min1store,minmax1store,bestloss,\
        bestperc = main(dataset,num_epochs,parname,step,bn,actchoice,zstep)

        # Make histograms
        # First, lets plot the validation error and test errors


        x1 = [x for x in range(0,len(valcorrect))]
        x2 = [x for x in range(0,len(testcorrect))]


        if not os.path.exists(figsave):
            os.makedirs(figsave)
        else:
            shutil.rmtree(figsave)
            os.makedirs(figsave)

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

        # Also, make figures of z's

        zstore1 = np.array(zstore1)
        zstore3 = np.array(zstore3)
        min1store = np.array(min1store)
        minmax1store = np.array(minmax1store)

        # For each z layered neuron stored...
        mycolors = ['b', 'g', 'r', 'm', 'k','c']
        if a ==0:
            mylabels = ['Sigmoid', 'Tanh', 'RELUCap', 'InvAbs', 'SoftPlus','ExpLin']
        elif a==1:
            mylabels = ['relu1cap', 'relu2cap', 'relu3cap', 'relu4cap', 'relu5cap']
        else:
            mylabels = ['reluRight','reluLeft']

        poollabels = ['Max','Sum']
        # Need to do three ensted for-loop inside for each zstore
        for i in range(len(zstore1[0,0,:])):

            s1 = pngcount
            s2 = pngcount+1
            s3 = pngcount+2
            s4 = pngcount+3
            plt.figure(s1)
            for j in range(len(zstore1[0,:,0])):
                plt.plot(zstore1[:,j,i],color=mycolors[j],label = mylabels[j],linewidth=2.0)
                plt.title(titlestring)
                plt.ylabel('Z Value')
                plt.xlabel('Epochs')
                plt.ylim(0,1)
            savestring = figsave + 'ZValuesforLayer1_' + str(i) + '.png'
            titlestring = 'Z Values for Layer 1'
            plt.title(titlestring)
            plt.legend(loc=1)
            plt.savefig(savestring)
            plt.close()

            plt.figure(s2)
            for j in range(len(zstore3[0,:,0])):
                plt.plot(zstore3[:, j, i], color=mycolors[j],label=mylabels[j],linewidth=2.0)
                plt.title(titlestring)
                plt.ylabel('Z Value')
                plt.xlabel('Epochs')
                plt.ylim(0,1)
            savestring = figsave + 'ZValuesforLayer3_' + str(i) + '.png'
            titlestring = 'Z Values for Layer 3'
            plt.title(titlestring)
            plt.legend(loc=1)
            plt.savefig(savestring)
            plt.close()

            plt.figure(s3)
            for j in range(len(min1store[0,:,0])):
                plt.plot(min1store[:, j, i], color=mycolors[j],label=mylabels[j],linewidth=2.0)
                plt.title(titlestring)
                plt.ylabel('Difference Value')
                plt.xlabel('Epochs')
            savestring = figsave + 'MinValuesforLayer1_' + str(i) + '.png'
            titlestring = 'Difference Between Running Min and Eta'
            plt.title(titlestring)
            plt.legend(loc=1)
            plt.savefig(savestring)
            plt.close()


            plt.figure(s4)
            for j in range(len(minmax1store[0,:,0])):
                plt.plot(minmax1store[:, j, i], color=mycolors[j],label=mylabels[j],linewidth=2.0)
                plt.title(titlestring)
                plt.ylabel('Difference Value')
                plt.xlabel('Epochs')
            savestring = figsave + 'MinMaxValuesforLayer1_' + str(i) + '.png'
            titlestring = 'Difference Between Running (Max-Min) and Delta'
            plt.title(titlestring)
            plt.legend(loc=1)
            plt.savefig(savestring)
            plt.close()




            pngcount+=4

        # One last thing I want to check is a few sum values to see how they change (hopefully not much at all)
        mylabels2 = ['sum1', 'sum2', 'sum3', 'sum4', 'sum5','sum6']
        for j in range(len(mycolors)):
            tempsum = np.zeros(len(zstore1))
            for i in range(len(zstore1)):

                tempsum[i] = np.sum(zstore1[i,:,j])
            plt.figure(pngcount)
            plt.plot(tempsum,color=mycolors[j],label=mylabels2[j])
            plt.ylabel('Epoch')
            plt.ylabel('Z Sums')
        savestring = figsave + 'ZSums.png'
        plt.title('A Few Sums')
        plt.ylim(0.8,1.2)
        plt.legend(loc=0)
        plt.savefig(savestring)
        plt.close()

        pngcount+=1

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


        storeacc[t] = testcorrect[0]
        storeloss[t] = testloss[0]

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

