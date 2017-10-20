#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.
This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html
More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import os
import time

import numpy as np
import theano
import theano.tensor as T
import lasagne
import pickle as pkl

from lasagne.layers import batch_norm
from lasagne.regularization import regularize_layer_params, l2, l1
import activations as act



def loadconvdata(dataset):

    # Load the dataset
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
    f.close()

    return [train_set, valid_set, test_set]

def build_nin(input_var=None):

    # Input layer, as usual:
    initial = lasagne.layers.InputLayer(shape=(None,3,32,32),
                                        input_var=input_var)


    conv1 = lasagne.layers.Conv2DLayer(
        incoming=initial,
        num_filters=192,
        filter_size=(5,5),pad = 2,nonlinearity=T.nnet.relu)


    ccp1 = lasagne.layers.NINLayer(incoming=conv1,num_units=160,nonlinearity=T.nnet.relu)

    ccp2 = lasagne.layers.NINLayer(incoming=ccp1,num_units=96,nonlinearity=T.nnet.relu)

    pool1 = lasagne.layers.MaxPool2DLayer(incoming = ccp2,pool_size=3,stride =2)

    drop1 = lasagne.layers.DropoutLayer(incoming=pool1,p=0.5)

    conv2 = lasagne.layers.Conv2DLayer(incoming = drop1,
                                    num_filters=192,
                                    filter_size=(5,5),pad=2,nonlinearity=T.nnet.relu)
    ccp3 = lasagne.layers.NINLayer(incoming=conv2,
                                   num_units=192,
                                   nonlinearity=T.nnet.relu)
    ccp4 = lasagne.layers.NINLayer(incoming=ccp3,
                                    num_units=192,
                                    nonlinearity=T.nnet.relu)
    pool2 = lasagne.layers.MaxPool2DLayer(incoming=ccp4,pool_size=3,stride=2)

    drop2 = lasagne.layers.DropoutLayer(incoming=pool2,p=0.5)

    conv3 = lasagne.layers.Conv2DLayer(incoming=drop2,
                                       num_filters=192,
                                       filter_size=3,
                                       pad=1,
                                       nonlinearity=T.nnet.relu)
    ccp5 = lasagne.layers.NINLayer(incoming=conv3,
                                   num_units=192,
                                   nonlinearity=T.nnet.relu)
    ccp6 = lasagne.layers.NINLayer(incoming=ccp5,
                                   num_units=100,
                                   nonlinearity=T.nnet.relu)
    pool3 = lasagne.layers.GlobalPoolLayer(incoming=ccp6)


    out = lasagne.layers.NonlinearityLayer(
            pool3,
            nonlinearity=lasagne.nonlinearities.softmax)

    return out

def build_cnn_bn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    #####################3
    # I need to remember to change the nonlinearities as well!!!!

    # Input layer, as usual:
    initial = lasagne.layers.InputLayer(shape=(None,3,32,32),
                                        input_var=input_var)

    drop1 = lasagne.layers.DropoutLayer(initial,p=0.2)

    conv1 = lasagne.layers.Conv2DLayer(
        incoming=drop1,
        num_filters=96,
        filter_size=(3,3),nonlinearity=T.nnet.relu)

    conv2 = lasagne.layers.Conv2DLayer(
        incoming=conv1,
        num_filters=96,
        filter_size=(3,3),nonlinearity=T.nnet.relu)

    conv3 = lasagne.layers.Conv2DLayer(
        incoming=conv2,
        num_filters=96,
        filter_size=(3,3),stride =2,nonlinearity=T.nnet.relu)

    drop2 = lasagne.layers.DropoutLayer(conv3,p=0.5)

    conv4 = lasagne.layers.Conv2DLayer(
        incoming=drop2,
        num_filters=192,
        filter_size=(3, 3), nonlinearity=T.nnet.relu)

    conv5 = lasagne.layers.Conv2DLayer(
        incoming=conv4,
        num_filters=192,
        filter_size=(3, 3), nonlinearity=T.nnet.relu)

    conv6 = lasagne.layers.Conv2DLayer(
        incoming=conv5,
        num_filters=192,
        filter_size=(3, 3), stride=2, nonlinearity=T.nnet.relu)

    drop3 = lasagne.layers.DropoutLayer(conv6,p=0.5)

    conv7 = lasagne.layers.Conv2DLayer(
        incoming=drop3,
        num_filters=192,
        filter_size=(3, 3), nonlinearity=T.nnet.relu)

    conv8 = lasagne.layers.Conv2DLayer(
        incoming=conv7,
        num_filters=192,
        filter_size=(1, 1), nonlinearity=T.nnet.relu)

    conv9 = lasagne.layers.Conv2DLayer(
        incoming=conv8,
        num_filters=100,
        filter_size=(1, 1), nonlinearity=T.nnet.relu)

    pool1 = lasagne.layers.GlobalPoolLayer(incoming = conv9)

    out = lasagne.layers.NonlinearityLayer(
            pool1,
            nonlinearity=lasagne.nonlinearities.softmax)

    return conv9,out


def build_mycnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    #####################3
    # I need to remember to change the nonlinearities as well!!!!

    # Input layer, as usual:
    initial = lasagne.layers.InputLayer(shape=(None,1,28,28),
                                        input_var=input_var)


    conv1 = batch_norm(lasagne.layers.Conv2DLayer(
        incoming=initial,
        num_filters=32,
        filter_size=(3,3),nonlinearity=T.nnet.relu))

    pool1 = lasagne.layers.MaxPool2DLayer(incoming=conv1,pool_size=(2, 2))

    conv2 = batch_norm(lasagne.layers.Conv2DLayer(
        incoming=pool1,
        num_filters=32,
        filter_size=(3,3),nonlinearity=T.nnet.relu))

    pool2 = lasagne.layers.MaxPool2DLayer(incoming=conv2,pool_size=(2, 2))


    feed1 = lasagne.layers.DenseLayer(incoming=pool2,
                                      num_units=400,
                                      nonlinearity=T.nnet.relu)



    out = lasagne.layers.DenseLayer(
            feed1,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)


    return out


def iterate_minibatches(inputs, targets, batchsize, train=False,shuffle=False):

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        if train==True:
            # Random Crop (only for cifar-100 for now).
            rb = np.random.randint(0,9)
            cb = np.random.randint(0,9)
            inp = inputs[excerpt,:,rb:(rb+32),cb:(cb+32)]

            # Horizontal flipping or not...
            if np.random.rand()>=0.5:
                inp = inp[:,:,::-1,:]

        else:
            inp = inputs[excerpt]

        yield inp, targets[excerpt]

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(dataset,num_epochs,parname,bn):
    # Load the dataset

    print("Loading data...")

    datasets = loadconvdata(dataset)

    X_train, y_train = datasets[0]
    X_val, y_val = datasets[1]
    X_test, y_test = datasets[2]

    nx_train = np.concatenate((X_train,X_val),axis=0)
    ny_train = np.concatenate((y_train,y_val),axis=0)

    print(nx_train.shape)

    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    ny_train = ny_train.astype(np.uint8)

    # Prepare Theano variables for inputs and targets

    input_var = T.ftensor4('inputs')
    target_var = T.ivector('targets')


    print("Building model...")

    # reg and build network
    l2param=0.0001
    network= build_mycnn(input_var)

    l2_penalty = regularize_layer_params(network, l2) * l2param
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network,deterministic=False)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    loss += l2_penalty
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss,params)
    #updates = lasagne.updates.adamax(loss,params,learning_rate=0.0001)
    #updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.001)
    #updates = lasagne.updates.sgd(loss,params,learning_rate=step)
    #updates = lasagne.updates.adadelta(loss, params)
    #updates = lasagne.updates.momentum(loss, params, learning_rate=lr, momentum=0.9)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_loss += l2_penalty
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    testloss = []
    valloss = []
    testcorrect = []
    valcorrect = []
    testpred = []
    valpred = []
    patience = 0
    epoch = 0
    its = 0
    while patience<50 and epoch < num_epochs and its <=165000:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train,  y_train, 128, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        temp = []
        for batch in iterate_minibatches(X_val, y_val, 128, shuffle=False):
            inputs, targets = batch
            err, acc,pred = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            its+=1

            if num_epochs-epoch<3 or patience>6:

                if not any(temp):
                    temp = pred.tolist()
                else:
                    temp = temp + pred.tolist()

        valloss.append(val_err/val_batches)
        valcorrect.append((val_acc/val_batches)*100)
        # Store predictions for later use
        if num_epochs-epoch<3 or patience >96:

            if not any(testpred):
                valpred = [temp]
            else:
                valpred = valpred + [temp]

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


        # This is checking if we need to stop training (overfitting)
        if epoch==0:
            bestloss = val_err/val_batches
        if epoch>=1:
            currentloss = val_err/val_batches
            if bestloss < currentloss:
                patience+=1
            else:
                bestloss = currentloss
                besttrain = train_err / train_batches
                bestperc = (val_acc/val_batches)*100
                valpred = [temp]
                if not os.path.exists(parname):
                    os.makedirs(parname)
                np.savez(parname, *lasagne.layers.get_all_param_values(network))
                patience=0

        epoch+=1

    lname = parname + '.npz'
    with np.load(lname) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)


    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    temp = []
    for batch in iterate_minibatches(X_test, y_test, 128, shuffle=False):
        inputs, targets = batch
        err, acc,pred = val_fn(inputs, targets)

        if not any(temp):
            temp = pred.tolist()
        else:
            temp = temp + pred.tolist()

        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    print("  val loss:\t\t\t{:.6f}".format(bestloss))
    print("  val accuracy:\t\t{:.2f} %".format(bestperc))

    testloss.append((test_err)/test_batches)

    testpred = temp

    testcorrect.append((test_acc/test_batches)*100)



    return testloss,testcorrect,testpred,valloss,valcorrect,valpred[-1],bestloss,bestperc

