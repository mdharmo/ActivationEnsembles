# Author: Mark Harmon
# Purpose: Preprocess the stl-10 dataset
from __future__ import division
import pickle as pkl
import numpy as np
from scipy import linalg
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


# mix up the temp data
vec = np.random.permutation(len(temp))
temp = temp[vec]
savespot = '/home/mharmon/ZProject/Data/stlfinal/'
beg = 0
for i in range(359):

    name = savespot + 'trainbatch_' + str(i) + '.p'
    # Get some data fo shizzle main
    end = beg+256
    X = np.copy(temp[beg:end])

    # Reshape the data
    X = X/255.
    X = np.reshape(X,(len(X),3*96*96))
    mean = X.mean(axis=1)
    X = X - mean[:,np.newaxis]

    X = np.reshape(X,(len(X),3,96,96)).astype('float32')

    pkl.dump([X,mean,norm],open(name,'wb'))

    beg = end

    print(end)


# Now do validation set

savespot = '/home/mharmon/ZProject/Data/stlfinal/'

for i in range(31):

    name = savespot + 'valbatch_' + str(i) + '.p'
    # Get some data fo shizzle main
    end = beg+256
    X = np.copy(temp[beg:end])

    # Reshape the data
    X = X/255.
    X = np.reshape(X,(len(X),3*96*96))
    mean = X.mean(axis=1)
    X = X - mean[:,np.newaxis]

    X = np.reshape(X,(len(X),3,96,96)).astype('float32')

    pkl.dump([X,mean,norm],open(name,'wb'))

    beg = end

    print(end)


# Now we have to do the test set...
del temp
savespot = '/home/mharmon/ZProject/Data/stlfinal/'

for i in range(59):
    name = savespot + 'testbatch_' + str(i) + '.p'
    # Get some data fo shizzle main
    end = beg + 256
    X = np.copy(temp[beg:end])

    # Reshape the data
    X = X/255.
    X = np.reshape(X,(len(X),3*96*96))
    mean = X.mean(axis=1)
    X = X - mean[:,np.newaxis]

    X = np.reshape(X, (len(X), 3, 96, 96)).astype('float32')

    pkl.dump([X,mean,norm], open(name, 'wb'))

    beg = end

    print(end)

name = savespot+ 'testbatch' + str(59)+ '.p'

X = np.copy(temp[beg:])

# Reshape the data
X = np.reshape(X, (len(X), 3 * 96 * 96))
mean = X.mean(axis=1)
X = X - mean[:, np.newaxis]
norm = np.sqrt(X.var(axis=1, ddof=1))
X = X / norm[:, np.newaxis]

X = np.reshape(X, (len(X), 3, 96, 96))

pkl.dump([X,mean,norm], open(name, 'wb'))

