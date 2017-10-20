# Author: Mark Harmon
# Purpose: Preprocess the cifar-100 data
from __future__ import division
import pickle as pkl
import numpy as np

import numpy as np
from scipy import linalg

# Do training data first
fo = open('/home/data/cifar-100-python/train','rb')
dict = pkl.load(fo,encoding='latin1')
#fo = open('/home/mdharmo/PycharmProjects/ZProject/data/cifar-100-python/train','rb')
#dict = pkl.load(fo)

# extract the data from the dictionary

X = np.array(dict['data'])
labels = np.array(dict['fine_labels'],'int32')


# reshape the data to be the proper shape for loading into my neural network
mean = X.mean(axis=1)
X = X - mean[:,np.newaxis]
norm = np.sqrt(X.var(axis=1,ddof=1))
X = X/norm[:,np.newaxis]

# ZCA whitening done here
X = (X - X.mean(axis=0)) # Center first
Cov = (np.dot(X.T,X) / (X.shape[0] + 0.1))
eigs, eigv = linalg.eigh(Cov)
sqrt_eigs = np.sqrt(eigs)
P_ = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)

X = np.dot(X,P_)



# Extract validation and training sets..
val = np.copy(X[40000:])
train = np.copy(X[:40000])
trainlab = np.copy(labels[:40000])
vallab = np.copy(labels[40000:])
del X
del labels



'''
ftrain = np.zeros((len(train),3,40,40))
train = np.reshape(train,(len(train),3,32,32)).astype('float32')

ftrain[:,:,4:36,4:36] = np.copy(train)
del train
ftrain = ftrain.astype('float32')
'''
train = np.reshape(train,(len(train),3,32,32)).astype('float32')
ftrain = train.astype('float32')


val = np.reshape(val,(len(val),3,32,32)).astype('float32')

# Finally, we do the same with the test set

fo = open('/home/data/cifar-100-python/test','rb')
dict = pkl.load(fo,encoding='latin1')

# extract the data from the dictionary

temp = dict['data']
testlabels = dict['fine_labels']

# reshape the data to be the proper shape for loading into my neural network

testdata = temp
testlabels = np.array(testlabels,'int32')

# reshape the data to be the proper shape for loading into my neural network
mean = testdata.mean(axis=1)
testdata = testdata - mean[:,np.newaxis]
norm = np.sqrt(testdata.var(axis=1,ddof=1))
testdata = testdata/norm[:,np.newaxis]

# ZCA whitening done here
testdata = (testdata - testdata.mean(axis=0)) # Center first
#Cov = (np.dot(testdata.T,testdata) / (testdata.shape[0] + 0.1))
#eigs, eigv = linalg.eigh(Cov)
#eigs = np.abs(eigs)
#sqrt_eigs = np.sqrt(eigs)

P_ = np.dot(eigv * (1.0 / sqrt_eigs), eigv.T)

testdata = np.dot(testdata,P_)
testdata = np.reshape(testdata,(len(testdata),3,32,32)).astype('float32')

alldata = [[ftrain,trainlab],[val,vallab],[testdata,testlabels]]

pkl.dump(alldata,open('/home/mharmon/ZProject/Data/cifarfinal.p','wb'))
