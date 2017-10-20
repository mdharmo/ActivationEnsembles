# Author: Mark Harmon
# Purpose: Preprocess the cifar-100 data
from __future__ import division
import pickle as pkl
import numpy as np

import numpy as np
from scipy import linalg
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

class ZCA(BaseEstimator, TransformerMixin):


    def __init__(self, regularization=10 ** -5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X):
        X = as_float_array(X, copy=self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T, X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1 / np.sqrt(S + self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)

        return X_transformed


# Do training data first
#fo = open('/home/mharmon/ZProject/Data/cifar-100-python/train','rb')
#dict = pkl.load(fo,encoding='latin1')
fo = open('/home/mdharmo/PycharmProjects/ZProject/data/cifar-100-python/train','rb')
dict = pkl.load(fo)

# extract the data from the dictionary

temp = dict['data']
labels = dict['fine_labels']

# reshape the data to be the proper shape for loading into my neural network

data = np.reshape(temp,(len(temp),3,32,32),'float32')
labels = np.array(labels,'int32')

#vec = np.random.permutation(len(data))
#data = data[vec]
#labels = labels[vec]

# Extract validation and training sets..
val = np.copy(data[40000:])
train = np.copy(data[0:40000])
trainlab = np.copy(labels[0:40000])
vallab = np.copy(labels[40000:])
del data
del labels

# Now we need to subtract the mean of each example and divide by the standard deviation.
# (Global Contrast Normalization)\
pixel_mean = np.mean(train,axis=0)
pixel_std = np.std(train,axis=0)
train = (train- pixel_mean)/pixel_std
val = (val- pixel_mean)/pixel_std


# Now we do some ZCA Whitening on the dataset.
# First flatten the data.  Whiten, then show the picture of the new data.
print('Whitening')
flat = np.reshape(train,(len(train),3*32*32))
trf = ZCA().fit(flat)
whitetrain = trf.transform(flat)
whitetrain = np.reshape(whitetrain,(len(whitetrain),3,32,32))
print('Done with Whitening')


# Now whiten the validation set
flat = np.reshape(val,(len(val),3*32*32))
whiteval = trf.transform(flat)
whiteval = np.reshape(whiteval,(len(whiteval),3,32,32))

# Make sure we have the correct data type
whitetrain = np.array(whitetrain,'float32')
whiteval = np.array(whiteval,'float32')


# Finally, we do the same with the test set

fo = open('/home/mharmon/ZProject/Data/cifar-100-python/test','rb')
dict = pkl.load(fo,encoding='latin1')

# extract the data from the dictionary

temp = dict['data']
testlabels = dict['fine_labels']

# reshape the data to be the proper shape for loading into my neural network

testdata = np.reshape(temp,(len(temp),3,32,32))
testlabels = np.array(testlabels,'int32')


# Use mean from before
testdata = (testdata - pixel_mean)/pixel_std
whitetest = testdata
# Do zca whitening
flat = np.reshape(whitetest,(len(testdata),3*32*32))
whitetest = trf.transform(flat)
whitetest = np.reshape(whitetest,(len(whitetest),3,32,32))


whitetest = np.array(whitetest,'float32')

alldata = [[whitetrain,trainlab],[whiteval,vallab],[whitetest,testlabels]]

pkl.dump(alldata,open('/home/mharmon/ZProject/Data/cifarwhite.p','wb'))

