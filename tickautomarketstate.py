# Author: Mark Harmon
# Purpose: Create autoencoder that can help deduce the current state of the market and identify changes within
# the market
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation,LSTM,Reshape,RepeatVector,Permute,Flatten,Input
from keras.layers.convolutional import Conv2D,UpSampling2D
from keras.layers.merge import Multiply
from keras.layers.wrappers import TimeDistributed
import pickle as pkl
import numpy as np
import sys
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
import time
from keras.layers.convolutional_recurrent import ConvLSTM2D

def build_ae(seqsize,stocks):
    # Make my autoencoder here
    # Pretty sure that I need to use the model api to make this work...

    main_input = Input(shape=(4,5,seqsize,1), dtype='float32', name='main_input')

    # This is where we do attention (This may need to change, attention for conv models seems odd).
    a1 = Reshape((4,5*seqsize))(main_input)
    a2 = TimeDistributed(Dense(1, activation='tanh'))(a1)
    a3 = Flatten()(a2)
    a4 = Activation('softmax')(a3)
    a5 = RepeatVector(5*seqsize)(a4)
    a6 = Permute([2,1])(a5)
    a7 = Reshape((4,5,seqsize,1))(a6)
    att = Multiply()([main_input,a7])

    # Encoder first
    conv1 = ConvLSTM2D(filters=64,kernel_size=(1,3),return_sequences=False,padding='same')(att)
    bn1 = BatchNormalization()(conv1)

    bottle_neck=1
    btlenk = Conv2D(filters=bottle_neck,kernel_size=(1,6))(bn1)
    bn2 = BatchNormalization()(btlenk)

    # Now upsample back up
    us = UpSampling2D(size=(1,2))(bn2)

    bef = Flatten()(us)
    rep = RepeatVector(4)(bef)
    rep2 = Reshape((4,5,seqsize,1))(rep)

    conv2 = ConvLSTM2D(64,return_sequences=True,kernel_size=(1,3),padding='same')(rep2)
    bn3 = BatchNormalization()(conv2)

    out = ConvLSTM2D(filters=1,kernel_size=(1,1),return_sequences=True,activation='linear')(bn3)

    model = Model(main_input,out)
    model.compile(optimizer='adadelta', loss='mean_squared_error',metrics=['accuracy'])

    return model

def main(weekcount,ticknum,winnum):
    stocks = 5
    mainadd = '/home/mharmon/FinanceProject/ModelResults/ae'+str(ticknum) + 'win' + str(winnum)
    address  = '/home/mharmon/FinanceProject/Data/tickdata/train'+str(ticknum) + 'cnn0.pkl'
    batchsize=256
    data,labels,dates = pkl.load(open(address,'rb'))
    seqsize = ticknum
    model = build_ae(seqsize,stocks)
    month_len = int(8064)
    week = int((month_len)/4.)
    data = np.swapaxes(data,2,4)
    data = np.swapaxes(data,2,3)
    # Need to additionally fit the smotes for training purposes...

    num_tests = 30
    for k in range(num_tests):
        beg = int((2*k)*month_len)
        end = int((2*(k+1)*month_len))


        # Load model if one is availabl


        modelsavepath = mainadd + '/Models/tickmodel' + str(0) + '.h5'
        epochx = data[beg:end]


        vallen = int(len(epochx) * 0.15)

        trainx = epochx[0:len(epochx) - vallen]

        valx = epochx[len(epochx) - vallen:]


        best_val = 100
        patience = 0
        while patience<5:
            firsttime = time.time()
            hist = model.fit(trainx, trainx, batch_size=batchsize,verbose=1, epochs=1, validation_data=(valx,valx))
            endtime = time.time()

            current_val = hist.history['val_loss'][0]
            print('')
            print('Window ' + str(0))
            print('Epoch Took %.3f Seconds' % (endtime - firsttime))
            print('Train Loss is ' + str(hist.history['loss'][0]))
            print('Validation Loss is ' + str(hist.history['val_loss'][0]))

            if np.mean(current_val)<best_val:
                best_val = np.mean(current_val)
                patience = 0
                model.save(modelsavepath)
                print('New Saved Model')

                # Save model
            else:
                patience+=1

            del model
            model = load_model(modelsavepath)


        hour_len = 12
        my_range = int(2*month_len/12.)
        validation_storage = []
        pngcount = 0
        first = beg
        last = first+hour_len
        x = range(my_range)
        for i in range(my_range):
            testx = data[first:last]


            hist = model.evaluate(testx,testx, verbose=0)
            validation_storage.append(hist[0])



            # I should only have 25 histograms for all test runs
            print()
            print('Finished Window ' + str(i))
            print('Validation Loss is ' + '%.5f' % hist[0])

            first+=hour_len
            last+=hour_len


        fig_save = mainadd + '/Figures/Val' + str(pngcount) +'Third'+'.png'
        pngcount+=1
        plt.figure(pngcount)
        plt.plot(x,validation_storage,linewidth=2)
        plt.title('Error by Hour')
        plt.xlabel('Hour')
        plt.ylabel('Loss')
        plt.savefig(fig_save)
        plt.close()


        # Do another test

    return

if __name__ == '__main__':

    weekcount = int(sys.argv[1])
    ticknum = int(sys.argv[2])
    winnum = int(sys.argv[3])

    main(weekcount,ticknum,winnum)