# Author: Mark Harmon
# Purpose: Preprocess the stl-10 dataset
from __future__ import division
import pickle as pkl
import numpy as np
from scipy import linalg
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

# Do training data first
f = open('/home/mharmon/ZProject/Data/stl.p','rb')

temp = pkl.load(f)

iterations = int(np.floor(len(temp)/256))

# mix up the temp data
vec = np.random.permutation(len(temp))
temp = temp[vec]
savespot = '/home/mharmon/ZProject/Data/stltrain/'
beg = 0
for i in range(273):

    name = savespot + 'trainbatch_' + str(i) + '.p'
    # Get some data fo shizzle main
    end = beg+256
    X = np.copy(temp[beg:end])

    # Reshape the data
    X = X/255.
    # Reshape the data
    X = np.reshape(X,(len(X),3*96*96))
    mean = X.mean(axis=1)
    X = X - mean[:,np.newaxis]

    X = np.reshape(X,(len(X),3,96,96)).astype('float32')

    pkl.dump([X,mean],open(name,'wb'))

    beg = end

    print(end)


# Now do validation set

savespot = '/home/mharmon/ZProject/Data/stlval/'

for i in range(58):

    name = savespot + 'valbatch_' + str(i) + '.p'
    # Get some data fo shizzle main
    end = beg+256
    X = np.copy(temp[beg:end])

    X = X/255.
    # Reshape the data
    X = np.reshape(X,(len(X),3*96*96))
    mean = X.mean(axis=1)
    X = X - mean[:,np.newaxis]

    X = np.reshape(X,(len(X),3,96,96)).astype('float32')

    pkl.dump([X,mean,norm],open(name,'wb'))

    beg = end

    print(end)

savespot = '/home/mharmon/ZProject/Data/stltest/'

for i in range(59):
    name = savespot + 'testbatch_' + str(i) + '.p'
    # Get some data fo shizzle main
    end = beg + 256
    X = np.copy(temp[beg:end])

    # Reshape the data
    X = X/255.
    # Reshape the data
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


