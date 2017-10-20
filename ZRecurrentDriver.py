# Author: Mark Harmon
# Purpose:  This is my mean driver code for running a feed-forward network
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu2')
import matplotlib
matplotlib.use('pdf')
from NNRecToy import main
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np

# Need to go through each data set.

dataset = '/home/mharmon/ZProject/Data/mnistpic.p'

bnvec = [False,True]
for b in range(2):

    bn = bnvec[b]

    if bn == False:
        bndir = '/NoBN'
    else:
        bndir = '/BN'

    for a in range(3):

        actsave = '/Act_' + str(a)
        actchoice = a
        num_epochs = 250

        pngcount=0


        step = 0.005
        zstep = 0.005
        save = '/home/mharmon/ZProject/ModelResults/mnist/Recurrent' + bndir + actsave

        figsave = save + '/Figures/'
        parname = save + '/Parameters'

        testloss,testcorrect,testpred,valloss,valcorrect,valpred,zstore1,zstore2,zstore3,bestloss,bestperc = main(dataset,num_epochs,parname,step,bn,actchoice,zstep)

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
        zstore2 = np.array(zstore2)
        zstore3 = np.array(zstore3)


        # For each z layered neuron stored...
        mycolors = ['b', 'g', 'r', 'm', 'k']
        if a ==0:
            mylabels = ['Sigmoid', 'Tanh', 'RELU', 'InvAbs', 'SoftPlus']
        elif a==1:
            mylabels = ['relu1', 'relu2', 'relu3', 'relu4', 'relu5']
        else:
            mylabels = ['reluRight','reluLeft']

        poollabels = ['Max','Sum']
        # Need to do three ensted for-loop inside for each zstore
        for i in range(len(zstore1[0,0,:])):

            s1 = pngcount
            s2 = pngcount+1
            s3 = pngcount+2
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
            for j in range(len(zstore2[0,:,0])):
                plt.plot(zstore2[:, j, i], color=mycolors[j],label=poollabels[j],linewidth=2.0)
                plt.title(titlestring)
                plt.ylabel('Z Value')
                plt.xlabel('Epochs')
                plt.ylim(0,1)
            savestring = figsave + 'ZValuesforLayer2_' + str(i) + '.png'
            titlestring = 'Z Values for Layer 2'
            plt.title(titlestring)
            plt.legend(loc=1)
            plt.savefig(savestring)
            plt.close()

            plt.figure(s3)
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




            pngcount+=3

        # One last thing I want to check is a few sum values to see how they change (hopefully not much at all)
        mylabels2 = ['sum1', 'sum2', 'sum3', 'sum4', 'sum5']
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

