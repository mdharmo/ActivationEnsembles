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

from ZLayer2 import ZLayer
from lasagne.layers import batch_norm
from lasagne.regularization import regularize_layer_params, l2, l1
from ZDense import ZDenseLayer
from ZDense import Zupdate
import activations as act
from lasagne.layers import BatchNormLayer
from MaxMinNorm import MaxMinNorm


def loadconvdata(dataset):

    # Load the dataset
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
    f.close()

    return [train_set, valid_set, test_set]


def build_z_bn(input_var=None,nonlins=lasagne.nonlinearities.rectify,mymins=0.0,mymaxes=1.0):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    #####################3
    # I need to remember to change the nonlinearities as well!!!!

    # Input layer, as usual:
    initial = lasagne.layers.InputLayer(shape=(None,784),
                                        input_var=input_var)


    feed1 = batch_norm(lasagne.layers.DenseLayer(
        incoming=initial,
        num_units=400,
        nonlinearity=None))
    my_units = [400]
    acts = [len(nonlins)]


    z1 = MaxMinNorm(incoming=feed1,z = lasagne.init.Constant(1./float(len(nonlins))),num_units=my_units[0],nonlinearity=nonlins)

    feed2 = batch_norm(lasagne.layers.DenseLayer(
        incoming=z1,
        num_units=400,
        nonlinearity=None))
    my_units += [400]
    acts += [len(nonlins)]


    z2 = MaxMinNorm(incoming=feed2,z = lasagne.init.Constant(1./float(len(nonlins))),num_units=my_units[1],nonlinearity=nonlins)

    feed3 = batch_norm(lasagne.layers.DenseLayer(
        incoming=z2,
        num_units=400,
        nonlinearity=None))
    my_units += [400]
    acts+= [len(nonlins)]


    z3 = MaxMinNorm(incoming=feed3,z = lasagne.init.Constant(1./float(len(nonlins))),num_units=my_units[2],nonlinearity=nonlins)

    out = lasagne.layers.DenseLayer(
            z3,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)


    return out,my_units,acts,z1,z2,z3

def build_z(input_var=None,nonlins=lasagne.nonlinearities.rectify):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    #####################3
    # I need to remember to change the nonlinearities as well!!!!

    # Input layer, as usual:
    initial = lasagne.layers.InputLayer(shape=(None,784),
                                        input_var=input_var)


    feed1 = lasagne.layers.DenseLayer(
        incoming=initial,
        num_units=400,
        nonlinearity=None)
    my_units = [feed1.num_units]
    acts = [len(nonlins)]
    z1 = MaxMinNorm(incoming=feed1, num_units=my_units[0], nonlinearity=nonlins)

    feed2 = lasagne.layers.DenseLayer(
        incoming=z1,
        num_units=400,
        nonlinearity=None)
    my_units += [feed2.num_units]
    acts+=[len(nonlins)]
    z2 = MaxMinNorm(incoming=feed2, num_units=my_units[0], nonlinearity=nonlins)


    feed3 = lasagne.layers.DenseLayer(
        incoming=z2,
        num_units=400,
        nonlinearity=None)
    my_units += [feed3.num_units]
    acts+=[len(nonlins)]
    z3 = MaxMinNorm(incoming=feed3, num_units=my_units[0], nonlinearity=nonlins)

    feed4 = lasagne.layers.DenseLayer(
        incoming=z3,
        num_units=400,
        nonlinearity=None)
    my_units += [feed4.num_units]
    acts+=[len(nonlins)]
    z4 = MaxMinNorm(incoming=feed4, num_units=my_units[0], nonlinearity=nonlins)

    feed5 = lasagne.layers.DenseLayer(
        incoming=z4,
        num_units=400,
        nonlinearity=None)
    my_units += [feed5.num_units]
    acts+=[len(nonlins)]
    z5 = MaxMinNorm(incoming=feed5, num_units=my_units[0], nonlinearity=nonlins)

    feed6 = lasagne.layers.DenseLayer(
        incoming=z5,
        num_units=400,
        nonlinearity=None)
    my_units += [feed6.num_units]
    acts+=[len(nonlins)]
    z6 = MaxMinNorm(incoming=feed6, num_units=my_units[0], nonlinearity=nonlins)

    feed7 = lasagne.layers.DenseLayer(
        incoming=z6,
        num_units=400,
        nonlinearity=None)
    my_units += [feed7.num_units]
    acts+=[len(nonlins)]
    z7 = MaxMinNorm(incoming=feed7, num_units=my_units[0], nonlinearity=nonlins)

    feed8 = lasagne.layers.DenseLayer(
        incoming=z7,
        num_units=400,
        nonlinearity=None)
    my_units += [feed8.num_units]
    acts+=[len(nonlins)]
    z8 = MaxMinNorm(incoming=feed8, num_units=my_units[0], nonlinearity=nonlins)

    feed9 = lasagne.layers.DenseLayer(
        incoming=z8,
        num_units=400,
        nonlinearity=None)
    my_units += [feed9.num_units]
    acts+=[len(nonlins)]
    z9 = MaxMinNorm(incoming=feed9, num_units=my_units[0], nonlinearity=nonlins)

    feed10 = lasagne.layers.DenseLayer(
        incoming=z9,
        num_units=400,
        nonlinearity=None)
    my_units += [feed10.num_units]
    acts+=[len(nonlins)]
    z10 = MaxMinNorm(incoming=feed10, num_units=my_units[0], nonlinearity=nonlins)

    out = lasagne.layers.DenseLayer(
            z10,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)


    return out,my_units,acts,z1,z2,z3

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):

    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(dataset,num_epochs,parname,step,bn,actchoice,zstep,seed):
    # Load the dataset

    print("Loading data...")
    datasets = loadconvdata(dataset)
    np.random.seed(seed)
    X_train, y_train = datasets[0]
    X_val, y_val = datasets[1]
    X_test, y_test = datasets[2]

    y_train = y_train.astype(np.uint8)
    y_val = y_val.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    nx_train = np.concatenate((X_train,X_val),axis=0)
    ny_train = np.concatenate((y_train,y_val),axis=0)


    # Prepare Theano variables for inputs and targets
    input_var = T.fmatrix('inputs')
    target_var = T.ivector('targets')


    print("Building model...")

    if actchoice ==0:
        nonlins = [T.nnet.sigmoid,act.n_tanh,act.n_relu,act.n_invabs,act.n_softrelu,act.n_explin]
        mymins = np.array([0.,-1.,0.,-1.,0.,-1.],dtype='float32')
        mymaxes = np.array([1.,1.,1.5,1.,1.5,1.5],dtype='float32')
    elif actchoice ==1:
        nonlins = [act.n_relu1, act.n_relu2, act.n_relu3, act.n_relu4, act.n_relu5]
        mymins = np.array([0.0,0.0,0.0,0.0,0.0],dtype='float32')
        mymaxes = np.array([1.0,1.0,1.0,1.0,1.0],dtype='float32')
    else:
        nonlins = [act.n_relu, act.n_reluleft]
        mymins = np.array([0.0,0.0,0.0],dtype='float32')
        mymaxes = np.array([1.0, 1.0,1.0],dtype='float32')


    # reg and build network
    l2param = 0
    network, my_units, acts, z1, z2,z3 = build_z_bn(input_var, nonlins,mymins,mymaxes)


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
    #updates = lasagne.updates.adadelta(loss,params)
    #updates = lasagne.updates.adamax(loss,params,learning_rate=0.0001)
    #updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.0001)
    updates = Zupdate(loss,params,acts,my_units,learning_rate=step)

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

    while patience<5 and epoch < num_epochs:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train,  y_train, 100, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        temp = []
        for batch in iterate_minibatches(X_val, y_val, 100, shuffle=False):
            inputs, targets = batch
            err, acc, pred = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

            if num_epochs-epoch<3 or patience>3:

                if not any(temp):
                    temp = pred.tolist()
                else:
                    temp = temp + pred.tolist()

        valloss.append(val_err/val_batches)
        valcorrect.append((val_acc/val_batches)*100)
        # Store predictions for later use
        if num_epochs-epoch<3 or patience >6:

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

        # Storage of z-values takes place here:

        if epoch ==0:
            zstore1 = z1.z.get_value()[:,0:20].reshape(1,len(nonlins),20).tolist()
            zstore2 = z2.z.get_value()[:,0:20].reshape(1,len(nonlins),20).tolist()
            zstore3 = z3.z.get_value()[:,0:20].reshape(1,len(nonlins),20).tolist()
            etastore1 = z1.eta.get_value()[:,0:20].tolist()
            deltastore1 = z1.delta.get_value()[:,0:20].tolist()
        else:
            zstore1+=z1.z.get_value()[:,0:20].reshape(1,len(nonlins),20).tolist()
            zstore2 += z2.z.get_value()[:, 0:20].reshape(1, len(nonlins), 20).tolist()
            zstore3 += z3.z.get_value()[:, 0:20].reshape(1, len(nonlins), 20).tolist()
            etastore1 += z1.eta.get_value()[:, 0:20].tolist()
            deltastore1 += z1.delta.get_value()[:, 0:20].tolist()

        # This is checking if we need to stop training (overfitting)
        if epoch==0:
            #bestloss = val_err/val_batches
            bestloss=val_acc/val_batches * 100
        if epoch>=1:
            currentloss = val_acc/val_batches*100

            if bestloss >= currentloss:
                patience+=1
            else:
                bestloss = currentloss
                bestperc = (val_acc/val_batches)*100
                besttrain = train_err/train_batches
                valpred = [temp]
                if not os.path.exists(parname):
                    os.makedirs(parname)
                np.savez(parname, *lasagne.layers.get_all_param_values(network))
                patience=0

        epoch+=1

        if (patience == 10) or (epoch == num_epochs):
            finallayer1zstore = z1.z.get_value()
            finallayer1zstore = np.reshape(finallayer1zstore,(len(finallayer1zstore),len(finallayer1zstore[0,:])))
            finallayer2zstore = z2.z.get_value()
            finallayer2zstore = np.reshape(finallayer2zstore, (len(finallayer2zstore), len(finallayer2zstore[0, :])))
            finallayer3zstore = z3.z.get_value()
            finallayer3zstore = np.reshape(finallayer3zstore, (len(finallayer3zstore), len(finallayer3zstore[0, :])))

    # Make sure we load the best model for testing
    lname = parname + '.npz'
    with np.load(lname) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)


    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    temp = []
    for batch in iterate_minibatches(X_test, y_test, 100, shuffle=False):
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
    # Optionally, you could now dump the network weights to a file like this :):

    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


    return testloss,testcorrect,testpred,valloss,valcorrect,valpred[-1],zstore1,zstore2,zstore3,etastore1,deltastore1,bestloss,bestperc,\
            finallayer1zstore,finallayer2zstore,finallayer3zstore

