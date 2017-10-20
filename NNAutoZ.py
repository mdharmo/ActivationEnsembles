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

from lasagne.regularization import regularize_layer_params, l2, l1
from os import listdir
from os.path import isfile, join
from lasagne.layers import batch_norm
from MaxMinNorm import MaxMinNorm
from ZDense import Zupdate
import activations as act

def build_cnn(input_var=None,nonlins = lasagne.nonlinearities.rectify):

    l_in = lasagne.layers.InputLayer(shape = (None, 3, 96, 96),input_var=input_var)
    e1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=5,pad='same',nonlinearity=None, W=lasagne.init.HeNormal(gain='relu'))

    my_units = [32]
    acts = [len(nonlins)]
    ez1 = MaxMinNorm(e1, num_units=32,z = lasagne.init.Constant(1/float(len(nonlins))), nonlinearity=nonlins)

    e2 = lasagne.layers.MaxPool2DLayer(ez1,pool_size=2)
    e3 = lasagne.layers.Conv2DLayer(e2,num_filters=64,filter_size=5,pad='same',nonlinearity=None, W=lasagne.init.HeNormal(gain='relu'))

    my_units += [64]
    acts += [len(nonlins)]
    ez2 = MaxMinNorm(e3, num_units=64,z = lasagne.init.Constant(1/float(len(nonlins))), nonlinearity=nonlins)

    e4 = lasagne.layers.MaxPool2DLayer(ez2,pool_size=2)
    e5 = lasagne.layers.Conv2DLayer(e4,num_filters=128,filter_size=5,pad='same',nonlinearity=None,W=lasagne.init.HeNormal(gain='relu'))

    my_units +=[128]
    acts +=[len(nonlins)]
    ez3 = MaxMinNorm(e5,num_units=128,z = lasagne.init.Constant(1/float(len(nonlins))),nonlinearity=nonlins)

    e6 = lasagne.layers.MaxPool2DLayer(ez3,pool_size=6)
    d6 = lasagne.layers.Upscale2DLayer(e6,scale_factor=6)

    d5 = lasagne.layers.TransposedConv2DLayer(d6, e5.input_shape[1],e5.filter_size, stride=e5.stride, crop=e5.pad,W=e5.W, flip_filters=not e5.flip_filters,nonlinearity=None,b=None)
    dz3 = MaxMinNorm(d5,num_units=64,nonlinearity=nonlins,z = ez2.z,eta=ez2.eta,delta=ez2.delta)


    d4 = lasagne.layers.Upscale2DLayer(dz3,scale_factor=2)
    d3 = lasagne.layers.TransposedConv2DLayer(d4, e3.input_shape[1],e3.filter_size, stride=e3.stride, crop=e3.pad,W=e3.W, flip_filters=not e3.flip_filters,nonlinearity=None,b=None)
    dz2 = MaxMinNorm(d3, num_units=32, nonlinearity=nonlins,z = ez1.z,eta=ez1.eta,delta=ez1.delta)

    d2 = lasagne.layers.Upscale2DLayer(dz2, scale_factor=2)
    out = lasagne.layers.TransposedConv2DLayer(d2, e1.input_shape[1], e1.filter_size, stride=e1.stride, crop=e1.pad,W=e1.W, flip_filters=not e1.flip_filters,nonlinearity=T.nnet.sigmoid,b=None)
    print(lasagne.layers.get_output_shape(out))

    return out,my_units,acts


def iterate_minibatches(mainset,filelist,sub, shuffle=False):


    filelen = len(filelist)-sub
    for start_idx in range(0, filelen):
        loadname = mainset + filelist[start_idx]
        inputs = pkl.load(open(loadname,'rb'))
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        else:
            indices = np.arange(len(inputs))
        yield inputs[indices].astype('float32'), inputs[indices].astype('float32')

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(set1,set2,set3,num_epochs,parname,storesave,actchoice):
    # Load the dataset
    trainlist = [f for f in listdir(set1) if isfile(join(set1, f))]
    vallist = [f for f in listdir(set2) if isfile(join(set2, f))]
    testlist = [f for f in listdir(set3) if isfile(join(set3, f))]

    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')
    output_var = T.ftensor4('outputs')

    # Create neural network model
    print("Building model and compiling functions...")
    if actchoice ==0:
        nonlins = [T.nnet.sigmoid,act.n_tanh,act.n_relu,act.n_invabs,act.n_softrelu,act.n_explin]
    elif actchoice ==1:
        nonlins = [act.n_relu1, act.n_relu2, act.n_relu3, act.n_relu4, act.n_relu5]
    else:
        nonlins = [act.n_relu,act.n_reluleft]

    # reg and build network
    print('Making Model...')
    l2param = 0.001
    network,my_units,acts = build_cnn(input_var=input_var,nonlins=nonlins)

    l2_penalty = regularize_layer_params(network, l2) * l2param

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network,deterministic=False)
    loss = lasagne.objectives.squared_error(prediction,output_var)
    loss = loss.mean()
    loss += l2_penalty
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    print(params)
    lr = theano.shared(np.array(1., dtype=theano.config.floatX))
    updates = Zupdate(loss, params, acts, my_units, learning_rate=lr, momentum=0.9)
    #updates = lasagne.updates.adadelta(loss,params)
    #updates = lasagne.updates.adamax(loss,params,learning_rate=0.0001)
    #updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.001)
    #updates = lasagne.updates.momentum(loss,params,learning_rate=step,momentum=0.9)
    #updates = lasagne.updates.sgd(loss,params,learning_rate=0.001)
    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,output_var)
    test_loss = test_loss.mean()
    test_loss += l2_penalty
    # As a bonus, also create an expression for the classification accuracy:

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var,output_var], [loss,prediction], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var,output_var], [test_loss, test_prediction])

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

    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))
    while patience<5 and epoch < num_epochs:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(set1,trainlist,0, shuffle=True):
            inputs,outputs = batch
            train_err_temp, pre = train_fn(inputs,outputs)
            train_err += train_err_temp
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        temp = []
        for batch in iterate_minibatches(set2,vallist,0, shuffle=False):
            inputs,outputs= batch
            err, pred = val_fn(inputs,outputs)
            val_err += err
            val_batches += 1

            if val_batches ==1:
                valsave = [pred,inputs]
            if num_epochs-epoch<3 or patience>3:

                if not any(temp):
                    temp = pred.tolist()
                else:
                    temp = temp + pred.tolist()

        valloss.append(val_err/val_batches)
        # Store predictions for later use
        if num_epochs-epoch<3 or patience >3:

            if not any(testpred):
                valpred = [temp]
            else:
                valpred = valpred + [temp]

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


        # This is checking if we need to stop training (overfitting)
        if epoch==0:
            bestloss = val_err/val_batches
        if epoch>=1:
            currentloss = val_err/val_batches
            if bestloss < currentloss:
                patience+=1
            else:
                bestloss = currentloss
                valpred = [temp]
                if not os.path.exists(parname):
                    os.makedirs(parname)
                np.savez(parname, *lasagne.layers.get_all_param_values(network))
                valsavefile = storesave + '/valstore.npy'

                np.save(valsavefile,valsave)
                patience=0

        epoch+=1

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
    loadname = set3 + testlist[0]
    testdata = pkl.load(open(loadname, 'rb'))
    beg = 0
    for batch in range(int(len(testdata)/100.)):
        end = beg+100
        inputs = testdata[beg:end].astype('float32')
        outputs = np.copy(inputs).astype('float32')
        err, pred = val_fn(inputs,outputs)

        if test_batches == 0:
            testsave = [pred,inputs]

        if not any(temp):
            temp = pred.tolist()
        else:
            temp = temp + pred.tolist()

        test_err += err
        test_batches += 1
        beg = end
    testsavefile = storesave + '/teststore.npy'
    np.save(testsavefile, testsave)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    print("  val loss:\t\t\t{:.6f}".format(bestloss))

    testloss.append((test_err)/test_batches)

    testpred = temp


    return testloss

