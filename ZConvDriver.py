# Author: Mark Harmon
# Purpose:  This is my mean driver code for running a feed-forward network
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1')
import matplotlib
matplotlib.use('pdf')
from NNConvZ import main
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import RegActs as racts
# Need to go through each data set.
matplotlib.rcParams.update({'font.size': 18})
dataset = '/home/mharmon/ZProject/Data/mnistpic.p'


numavg = 1
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
        actchoice = a
        num_epochs = 1500
        pngcount=0
        choose = 'conv'


        step = 1.0
        zstep = 0.001
        save = '/home/mharmon/ZProject/ModelResults/mnist/FinalZ' + bndir + actsave

        figsave = save + '/Figures/'
        parname = save + '/Parameters'

        testloss, testcorrect, testpred, valloss, valcorrect, valpred, zstore1, zstore2, zstore3, eta, delta, bestloss, \
        bestperc, finallayer1zstore, finallayer2zstore, finallayer3zstore = main(dataset, num_epochs, parname, step, bn,
                                                                                 actchoice, zstep)

        # Make histograms
        # First, lets plot the validation error and test errors


        x1 = [x for x in range(0, len(valcorrect))]
        x2 = [x for x in range(0, len(testcorrect))]

        if not os.path.exists(figsave):
            os.makedirs(figsave)
        else:
            shutil.rmtree(figsave)
            os.makedirs(figsave)

        name = figsave + 'Val_Error.png'
        plt.figure(pngcount, figsize=(8, 6))
        plt.plot(x1, valcorrect, linewidth=2.0)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        # plt.title('Validation Error Plot')
        plt.savefig(name, dpi=500)
        plt.close()
        pngcount += 1

        x3 = [x for x in range(0, len(valloss))]
        name = figsave + 'Cost.png'
        plt.figure(pngcount, figsize=(8, 6))
        plt.plot(x3, valloss, linewidth=2.0)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        # plt.title('Cost Plot (Validation Data)')
        plt.savefig(name, dpi=500)
        plt.close()
        pngcount += 1

        # Get total Histogram with all roles:
        bins = np.linspace(0, 1, 100)
        testpred = np.array(testpred)
        myargmax = np.argmax(testpred, axis=1)

        plottest = np.zeros(myargmax.shape)
        for i in range(len(myargmax)):
            plottest[i] = testpred[i, myargmax[i]]

        plt.figure(pngcount, figsize=(8, 6))
        # titlestring = 'Test Probability Histogram (Highest Probability)'
        savestring = figsave + 'HistogramAllClasses.png'
        plt.hist(plottest, bins)
        plt.xlabel('Values (Bins)')
        plt.ylabel('Frequency')
        plt.savefig(savestring, dpi=500)
        plt.close()
        pngcount += 1

        # Also, make figures of z's

        zstore1 = np.array(zstore1)
        zstore2 = np.array(zstore2)
        zstore3 = np.array(zstore3)

        # For each z layered neuron stored...
        mycolors = ['b', 'g', 'r', 'm', 'k', 'c']
        if a == 0:
            mylabels = ['Sigmoid', 'Tanh', 'RELU', 'InvAbs', 'SoftPlus', 'ExpLin']
        elif a == 1:
            mylabels = ['relu1', 'relu2', 'relu3', 'relu4', 'relu5']
        else:
            mylabels = ['reluRight', 'reluLeft']

        val2 = np.zeros((20, 20))
        val1 = np.zeros((20, 20))
        val3 = np.zeros((20, 20))
        # Need to do three ensted for-loop inside for each zstore
        # For each neuron....
        for i in range(len(zstore1[0, 0, :])):

            s1 = pngcount
            s2 = pngcount + 1
            s3 = pngcount + 2
            s4 = pngcount + 3
            s5 = pngcount + 4
            s6 = pngcount + 5

            plt.figure(s1, figsize=(8, 6))
            for j in range(len(zstore1[0, :, 0])):
                plt.plot(zstore1[:, j, i], color=mycolors[j], label=mylabels[j], linewidth=2.0)
                plt.ylabel(r'$ \alpha $ Value')
                plt.xlabel('Epochs')
                plt.ylim(0, 1)
            savestring = figsave + 'ZValuesforLayer1_' + str(i) + '.png'
            titlestring = 'Z Values for Layer 1'
            # plt.title(titlestring)
            plt.legend(loc=1)
            plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
            plt.close()

            plt.figure(s2, figsize=(8, 6))
            for j in range(len(zstore2[0, :, 0])):
                plt.plot(zstore2[:, j, i], color=mycolors[j], label=mylabels[j], linewidth=2.0)
                plt.ylabel(r'$ \alpha $ Value')
                plt.xlabel('Epochs')
                plt.ylim(0, 1)
            savestring = figsave + 'ZValuesforLayer2_' + str(i) + '.png'
            titlestring = 'Z Values for Layer 2'
            # plt.title(titlestring)
            plt.legend(loc=1)
            plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
            plt.close()

            plt.figure(s3, figsize=(8, 6))
            for j in range(len(zstore3[0, :, 0])):
                plt.plot(zstore3[:, j, i], color=mycolors[j], label=mylabels[j], linewidth=2.0)
                plt.title(titlestring)
                plt.ylabel(r'$ \alpha $ Value')
                plt.xlabel('Epochs')
                plt.ylim(0, 1)
            savestring = figsave + 'ZValuesforLayer3_' + str(i) + '.png'
            titlestring = 'Z Values for Layer 3'
            # plt.title(titlestring)
            plt.legend(loc=1)
            plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
            plt.close()

            if a == 0:
                nonlins = [racts.n_sig, racts.n_tanh, racts.n_relu, racts.n_invabs, racts.n_softrelu, racts.n_explin]
            elif a == 1:
                nonlins = [racts.n_relu1, racts.n_relu2, racts.n_relu3, racts.n_relu4, racts.n_relu5]
            else:
                nonlins = [racts.n_relu, racts.n_reluleft]

            vec = np.linspace(-1, 1, num=20)
            plt.figure(s4, figsize=(8, 6))
            newlabels = ['Neuron 1', 'Neuron 2', 'Neuron 3', 'Neuron 4', 'Neuron 5']
            for j in range(len(zstore1[0, :, 0])):
                val1[i, :] += zstore1[-1, j, i] * nonlins[j](vec)

            plt.plot(val1[i, :], linewidth=2.0)
            plt.ylabel('Function Value')
            plt.xlabel('X')
            savestring = figsave + 'FunkforLayer1_' + str(i) + '.png'
            titlestring = 'Example Functions Layer 1'
            # plt.title(titlestring)
            plt.legend(loc=2)

            plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
            plt.close()

            vec = np.linspace(-1, 1, num=20)
            plt.figure(s5, figsize=(8, 6))
            for j in range(len(zstore2[0, :, 0])):
                val2[i, :] += zstore2[-1, j, i] * nonlins[j](vec)

            plt.plot(val2[i, :], linewidth=2.0)
            plt.ylabel('Function Value')
            plt.xlabel('X')
            savestring = figsave + 'FunkforLayer2_' + str(i) + '.png'
            # titlestring = 'Example Functions Layer 2'
            plt.title(titlestring)
            plt.legend(loc=2)
            plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
            plt.close()

            vec = np.linspace(-1, 1, num=20)
            plt.figure(s6, figsize=(8, 6))
            for j in range(len(zstore2[0, :, 0])):
                val3[i, :] += zstore3[-1, j, i] * nonlins[j](vec)

            plt.plot(val3[i, :], linewidth=2.0)
            plt.ylabel('Function Value')
            plt.xlabel('X')
            savestring = figsave + 'FunkforLayer3_' + str(i) + '.png'
            # titlestring = 'Example Functions Layer 3'
            plt.title(titlestring)
            plt.legend(loc=2)
            plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
            plt.close()

            pngcount += 6

        mycolors = ['b', 'g', 'r', 'm', 'k', 'c']
        if a == 0:
            mylabels = ['Sigmoid', 'Tanh', 'RELU', 'InvAbs', 'SoftPlus', 'ExpLin']
        elif a == 1:
            mylabels = ['relu1', 'relu2', 'relu3', 'relu4', 'relu5']
        else:
            mylabels = ['reluRight', 'reluLeft']

        bins = np.linspace(0, 1, 100)
        plt.figure(pngcount)
        for j in range(len(finallayer1zstore)):
            plt.hist(finallayer1zstore[j, :], bins, alpha=0.4, label=mylabels[j])
            plt.axis([0,1,0,20])
        plt.ylabel('Count')
        plt.xlabel('Bins')
        savestring = figsave + 'HistogramLayer1.png'
        plt.legend(loc='upper right')
        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1

        plt.figure(pngcount)
        for j in range(len(finallayer2zstore)):
            plt.hist(finallayer2zstore[j, :], bins, alpha=0.4, label=mylabels[j])
            plt.axis([0, 1, 0, 20])
        plt.ylabel('Count')
        plt.xlabel('Bins')
        savestring = figsave + 'HistogramLayer2.png'
        plt.legend(loc='upper right')
        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1

        plt.figure(pngcount)
        for j in range(len(finallayer3zstore)):
            plt.hist(finallayer3zstore[j, :], bins, alpha=0.4, label=mylabels[j])
            plt.axis([0, 1, 0, 200])
        plt.ylabel('Count')
        plt.xlabel('Bins')
        savestring = figsave + 'HistogramLayer3.png'
        plt.legend(loc='upper right')
        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1

        if a + 1== 2:
            myloc = 9
        else:
            myloc = 4

        newlabels = ['Neuron 1', 'Neuron 2', 'Neuron 3', 'Neuron 4', 'Neuron 5']
        plt.figure(pngcount, figsize=(8, 6))
        for j in range(5):
            plt.plot(val1[j, :], linewidth=2.0, color=mycolors[j], label=newlabels[j])
            plt.ylabel('Function Value')
            plt.xlabel('X')
            savestring = figsave + 'FunksCompareLayer1' + '.png'
            titlestring = 'Example Functions Layer 1'
            plt.title(titlestring)
        plt.legend(loc=myloc)

        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1
        plt.figure(pngcount, figsize=(8, 6))
        for j in range(5):
            plt.plot(val2[j, :], linewidth=2.0, color=mycolors[j], label=newlabels[j])
            plt.ylabel('Function Value')
            plt.xlabel('X')
            savestring = figsave + 'FunkCompareLayer2' + '.png'
            titlestring = 'Example Functions Layer 2'
            plt.title(titlestring)
        plt.legend(loc=myloc)

        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1

        plt.figure(pngcount, figsize=(8, 6))
        for j in range(5):
            plt.plot(val2[j, :], linewidth=2.0, color=mycolors[j], label=newlabels[j])
            plt.ylabel('Function Value')
            plt.xlabel('X')
            savestring = figsave + 'FunkCompareLayer3' + '.png'
            # titlestring = 'Example Functions Layer 3'
            plt.title(titlestring)
        plt.legend(loc=myloc)

        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1

        # One last thing I want to check is a few sum values to see how they change (hopefully not much at all)
        mylabels2 = ['sum1', 'sum2', 'sum3', 'sum4', 'sum5', 'sum6']
        for j in range(len(mycolors)):
            tempsum = np.zeros(len(zstore1))
            for i in range(len(zstore1)):
                tempsum[i] = np.sum(zstore1[i, :, j])
            plt.figure(pngcount, figsize=(8, 6))
            plt.plot(tempsum, color=mycolors[j], label=mylabels2[j])
            plt.ylabel('Epoch')
            plt.ylabel('Z Sums')
        savestring = figsave + 'ZSums.png'
        # plt.title('A Few Sums')
        plt.ylim(0.8, 1.2)
        plt.legend(loc=0)
        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()

        pngcount += 1

        eta = np.array(eta)
        delta = np.array(delta)
        mylabseta = ['Eta 1', 'Eta 2', 'Eta 3', 'Eta 4', 'Eta 5']
        mylabsdelt = ['Delta 1', 'Delta 2', 'Delta 3', 'Delta 4', 'Delta 5']

        for j in range(5):
            plt.figure(pngcount, figsize=(8, 6))
            plt.plot(eta[:, j], linewidth=2.0, color=mycolors[j], label=mylabseta[j])

        plt.ylabel('Eta Value')
        plt.xlabel('Iteration')
        savestring = figsave + 'EtaLayer1_' + str(i) + '.png'
        titlestring = 'Example Eta Values'
        # plt.title(titlestring)
        plt.legend(loc=1)

        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1

        for j in range(5):
            plt.figure(pngcount, figsize=(8, 6))
            plt.plot(delta[:, j], linewidth=2.0, color=mycolors[j], label=mylabsdelt[j])

        plt.ylabel('Delta Value')
        plt.xlabel('Iteration')
        savestring = figsave + 'DeltaLayer1_' + str(i) + '.png'
        titlestring = 'Example Delta Values'
        # plt.title(titlestring)
        plt.legend(loc=1)

        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1

        # Here is where I make a table
        savestring = figsave + 'Table.png'
        cell_text = [[str(testloss[0]), str(testcorrect[0])], [str(bestloss), str(bestperc)]]
        rows = ['Test', 'Validation']
        cols = ['Loss', 'Accuracy']
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              colLabels=cols, loc=0)
        plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
        plt.close()
        pngcount += 1

        storeacc[t] = testcorrect[0]
        storeloss[t] = testloss[0]

    print('Final Averages:')
    print('Average Test Accuracy:\t\t\t {:.6f} %'.format(np.mean(storeacc)))
    print('Average Test Loss:\t\t\t {:.6f}'.format(np.mean(storeloss)))
    print('Min Test Accuracy:\t\t\t {:.6f} %'.format(np.min(storeacc)))
    print('Max Test Accuracy:\t\t\t {:.6f} %'.format(np.max(storeacc)))
    # Here is where I make a table
    savestring = figsave + 'TableFinal.png'
    cell_text = [[np.mean(storeloss), np.mean(storeacc), np.min(storeacc), np.max(storeacc)]]
    rows = ['Test']
    cols = ['Loss', 'Accuracy', 'Min Accuracy', 'Max Accuracy']
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          colLabels=cols, loc=0)
    plt.savefig(savestring, dpi=500,bbox_inches = 'tight')
    plt.close()
