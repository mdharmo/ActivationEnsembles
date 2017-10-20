import numpy as np
import pickle as pkl

dataset = '/home/mharmon/ZProject/Data/mnistpic.p'

# Load the dataset
f = open(dataset, 'rb')
train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
f.close()

traindat = np.array(train_set[0],'float32')
valdat = np.array(valid_set[0],'float32')
testdat=np.array(test_set[0],'float32')

for i in range(len(traindat)):

    traindat[i] = (traindat[i] - np.min(traindat[i]))/(np.max(traindat[i])-np.min(traindat[i]))

for i in range(len(valdat)):
    valdat[i] = (valdat[i] - np.min(valdat[i])) / (np.max(valdat[i]) - np.min(valdat[i]))

for i in range(len(testdat)):
    testdat[i] = (testdat[i] - np.min(testdat[i])) / (np.max(testdat[i]) - np.min(testdat[i]))

traindat = np.nan_to_num(traindat)
valdat = np.nan_to_num(valdat)
testdat = np.nan_to_num(testdat)

traindat = np.array(traindat,'float32')
valdat = np.array(valdat,'float32')
testdat = np.array(testdat,'float32')

train_set = [traindat,train_set[1]]
valid_set = [valdat,valid_set[1]]
test_set=[testdat,test_set[1]]


pkl.dump([train_set,valid_set,test_set],open(dataset,'wb'))

