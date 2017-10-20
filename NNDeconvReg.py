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
import activations as act



def loadconvdata(dataset):

    # Load the dataset
    f = open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
    f.close()

    return [train_set, valid_set, test_set]


def build_cnn_final(conv1,conv2,feed1,feed2,input_var=None,):

    # Now we put them all together for final training.
    initial1 = lasagne.layers.InputLayer(shape=(None, 3, 96, 96),
                                         input_var=input_var)
    fconv1 = lasagne.layers.Conv2DLayer(incoming=initial1,
                                        num_filters=64,
                                        filter_size=(3,3),pad=1,W=conv1.W)
    fpool1 = lasagne.layers.Maxpool2DLayer(fconv1,pool_size=2)
    fconv2 = lasagne.layers.Conv2DLayer(incoming=fpool1,
                                        num_filters=32,
                                        filters_size=(3,3),pad=1,W=conv2.W)

    fpool2 = lasagne.layers.MaxPool2DLayer(fconv2,pool_size=2)

    ffeed1 = lasagne.layers.DenseLayer(incoming=fpool2,num_units=400,W=feed1.W)

    out = lasagne.layers.Denselayer(incoming=ffeed1,num_units=10,nonlinearity=lasagne.nonlinearities.softmax,W=feed2.W)

def build_cnn_deconv(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.
    #####################3
    # I need to remember to change the nonlinearities as well!!!!

    # Input layer, as usual:

    # First auto-encoder layer
    initial1 = lasagne.layers.InputLayer(shape=(None,3,96,96),
                                        input_var=input_var)


    conv1 = lasagne.layers.Conv2DLayer(
        incoming=initial1,
        num_filters=64,
        filter_size=(3,3),pad=1,nonlinearity=act.n_threshrelu)

    pool1 = lasagne.layers.MaxPool2DLayer(conv1,pool_size=2)

    unpool1 = lasagne.layers.Upscale2DLayer(pool1,scale_factor = 2)

    deconv1 = lasagne.layers.TransposedConv2DLayer(unpool1,
                                                   num_filters=3,
                                                   filter_size = (3,3),
                                                   pad=1,nonlinearity=lasagne.nonlinearities.linear,W=conv1.W)


    # Second auto-encoder layer
    initial2 = lasagne.layers.InputLayer(shape=pool1.input_shape,
                                         input_var=input_var)

    conv2 = lasagne.layers.Conv2DLayer(incoming=initial2,
                                       num_filters=32,
                                       filters_size=(3,3),pad=1,nonlinearity=act.n_threshrelu)
    pool2 = lasagne.layers.MaxPool2DLayer(conv2,pool_size=2)

    unpool2 = lasagne.layers.Upscale2DLayer(pool2,scale_factor=2)

    deconv2 = lasagne.layers.TransposedConv2DLayer(unpool2,
                                                   num_filters=32,
                                                   filter_size=(3,3),
                                                   pad=1,nonlinearity=lasagne.nonlinearities.linear,W=conv2.W)

    # Third autoencoder layer
    initial3 = lasagne.layers.InputLayer(shape=pool2.input_shape,input_var=input_var)

    flat1 = lasagne.layers.FlattenLayer(incoming=initial3, outdim=2)

    feed1 = lasagne.layers.DenseLayer(incoming=flat1,
                                      num_units=400,nonlinearities = act.n_threshrelu)

    # Fourth autoencoder layer
    defeed1 = lasagne.layers.DenseLayer(incoming=feed1,num_units=feed1.input_shape[1],nonlinearity= lasagne.nonlinearities.linear)

    initial4 = lasagne.layers.InputLayer(shape=feed1.input_shape,input_var=input_var)

    feed2 = lasagne.layers.DenseLayer(incoming=initial4,num_units=10,nonlinearity=lasagne.nonlinearities.softmax)


    return pool1,pool2,feed1,deconv1,deconv2,defeed1,feed2,conv1,conv2


def iterate_minibatches(type):

    # load the minibatch used..

    if type == 'train':

        for its in range(273):
            batch = '/home/mharmon/ZProject/Data/stltrain/trainbatch_' + str(its) + '.p'
            inputs,dump,dump = pkl.load(open(batch,'rb'))

            yield inputs
    elif type == 'val':
        for its in range(58):
            batch = '/home/mharmon/ZProject/Data/stlval/valbatch_' + str(its) + '.p'
            inputs,dump,dump = pkl.load(open(batch,'rb'))

            yield inputs
    elif type == 'test':
        for its in range(60):
            batch = '/home/mharmon/ZProject/Data/stltest/testbatch_' + str(its) + '.p'
            inputs,dump,dump = pkl.load(open(batch,'rb'))

            yield inputs
    else:
        print('You fucked up!!')



# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(num_epochs,parname):


    # Prepare Theano variables for inputs and targets
    input_var = T.ftensor4('inputs')
    target_var = T.ftensor4('inputs')
    target_var2 = T.ivector('target')

    print("Building autoencoder models...")
    pool1, pool2, feed1, deconv1, deconv2, defeed1, feed2,conv1,conv2 = build_cnn_deconv(input_var)

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    # Build the first model here...
    prediction1 = lasagne.layers.get_output(deconv1,deterministic=False)
    loss1 = lasagne.objectives.squared_error(prediction1, target_var)
    loss1 = loss1.mean()
    params1 = lasagne.layers.get_all_params(pool1, trainable=True)
    updates1 = lasagne.updates.momentum(loss1, params1, learning_rate=0.01, momentum=0.9)
    test_prediction1 = lasagne.layers.get_output(deconv1, deterministic=True)
    test_loss1 = lasagne.objectives.squared_error(test_prediction1,target_var)
    test_loss1 = test_loss1.mean()
    train_fn1 = theano.function([input_var, target_var], loss1, updates=updates1)
    val_fn1 = theano.function([input_var, target_var], [test_loss1, test_prediction1])

    # Finally, launch the training loop.
    print("Starting training 1st auto...")
    # We iterate over epochs:
    valloss = []
    testpred = []
    valpred = []
    patience = 0
    epoch = 0
    its = 0
    while patience<50 and epoch < num_epochs or its >165000:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(type='train'):
            inputs = batch
            train_err += train_fn1(inputs, inputs)
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        temp = []
        for batch in iterate_minibatches( type='val'):
            inputs = batch
            err, pred = val_fn1(inputs, inputs)
            val_err += err
            val_batches += 1


            if num_epochs-epoch<3 or patience>6:

                if not any(temp):
                    temp = pred.tolist()
                else:
                    temp = temp + pred.tolist()

        valloss.append(val_err/val_batches)
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
                np.savez(parname, *lasagne.layers.get_all_param_values(pool1))
                patience=0

        epoch+=1

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    # Build the first model here...
    prediction2 = lasagne.layers.get_output(deconv2,deterministic=False)
    loss2 = lasagne.objectives.squared_error(prediction2, target_var)
    loss2 = loss2.mean()
    params2 = lasagne.layers.get_all_params(pool2, trainable=True)
    updates2 = lasagne.updates.momentum(loss1, params2, learning_rate=0.01, momentum=0.9)
    test_prediction2 = lasagne.layers.get_output(deconv2, deterministic=True)
    test_loss2 = lasagne.objectives.squared_error(test_prediction2,target_var)
    test_loss2 = test_loss2.mean()
    train_fn2 = theano.function([input_var, target_var], loss2, updates=updates2)
    val_fn2 = theano.function([input_var, target_var], [test_loss2, test_prediction2])

    # Finally, launch the training loop.
    print("Starting training 2nd auto...")
    # We iterate over epochs:
    valloss = []
    testpred = []
    valpred = []
    patience = 0
    epoch = 0
    its = 0
    while patience<50 and epoch < num_epochs or its >165000:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(type='train'):
            inputs = batch
            train_err += train_fn2(inputs, inputs)
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        temp = []
        for batch in iterate_minibatches( type='val'):
            inputs = batch
            err, pred = val_fn2(inputs, inputs)
            val_err += err
            val_batches += 1


            if num_epochs-epoch<3 or patience>6:

                if not any(temp):
                    temp = pred.tolist()
                else:
                    temp = temp + pred.tolist()

        valloss.append(val_err/val_batches)
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
                np.savez(parname, *lasagne.layers.get_all_param_values(pool1))
                patience=0

        epoch+=1

    ################################################################################################################
    ################################################################################################################
    ################################################################################################################

    # Build the first model here...
    prediction3 = lasagne.layers.get_output(defeed1,deterministic=False)
    loss3 = lasagne.objectives.squared_error(prediction3, target_var)
    loss3 = loss3.mean()
    params3 = lasagne.layers.get_all_params(defeed1, trainable=True)
    updates3 = lasagne.updates.momentum(loss3, params3, learning_rate=0.01, momentum=0.9)
    test_prediction3 = lasagne.layers.get_output(defeed1, deterministic=True)
    test_loss3 = lasagne.objectives.squared_error(test_prediction3,target_var)
    test_loss3 = test_loss3.mean()
    train_fn3 = theano.function([input_var, target_var], loss3, updates=updates3)
    val_fn3 = theano.function([input_var, target_var], [test_loss3, test_prediction3])

    # Finally, launch the training loop.
    print("Starting training for 3rd auto...")
    # We iterate over epochs:
    valloss = []
    testpred = []
    valpred = []
    patience = 0
    epoch = 0
    its = 0
    while patience<50 and epoch < num_epochs or its >165000:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(type='train'):
            inputs = batch
            train_err += train_fn3(inputs, inputs)
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        temp = []
        for batch in iterate_minibatches( type='val'):
            inputs = batch
            err, pred = val_fn3(inputs, inputs)
            val_err += err
            val_batches += 1


            if num_epochs-epoch<3 or patience>6:

                if not any(temp):
                    temp = pred.tolist()
                else:
                    temp = temp + pred.tolist()

        valloss.append(val_err/val_batches)
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
                np.savez(parname, *lasagne.layers.get_all_param_values(pool1))
                patience=0

        epoch+=1


    # Build the last part, which actually classifies here....
    prediction4 = lasagne.layers.get_output(feed2,deterministic=False)
    loss4 = lasagne.objectives.squared_error(prediction4, target_var2)
    loss4 = loss4.mean()
    params4 = lasagne.layers.get_all_params(feed2, trainable=True)
    updates4 = lasagne.updates.momentum(loss4, params4, learning_rate=0.01, momentum=0.9)
    test_prediction4 = lasagne.layers.get_output(feed2, deterministic=True)
    test_loss4 = lasagne.objectives.squared_error(test_prediction4,target_var2)
    test_loss4 = test_loss4.mean()
    train_fn4 = theano.function([input_var, target_var2], loss4, updates=updates4)
    val_fn4 = theano.function([input_var, target_var2], [test_loss4, test_prediction4])

    # Finally, launch the training loop.
    print("Starting training for 4th auto...")
    # We iterate over epochs:
    valloss = []
    testpred = []
    valpred = []
    patience = 0
    epoch = 0
    its = 0
    while patience<50 and epoch < num_epochs or its >165000:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(type='train'):
            inputs = batch
            train_err += train_fn4(inputs, inputs)
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        temp = []
        for batch in iterate_minibatches( type='val'):
            inputs = batch
            err, pred = val_fn4(inputs, inputs)
            val_err += err
            val_batches += 1


            if num_epochs-epoch<3 or patience>6:

                if not any(temp):
                    temp = pred.tolist()
                else:
                    temp = temp + pred.tolist()

        valloss.append(val_err/val_batches)
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
                np.savez(parname, *lasagne.layers.get_all_param_values(pool1))
                patience=0

        epoch+=1



    print("Building final model...")
    network = build_cnn_final(conv1,conv2,feed1,feed2,input_var)

    # Build the last part, which actually classifies here....
    prediction5 = lasagne.layers.get_output(network,deterministic=False)
    loss5 = lasagne.objectives.squared_error(prediction5, target_var2)
    loss5 = loss5.mean()
    params5 = lasagne.layers.get_all_params(network, trainable=True)
    updates5 = lasagne.updates.momentum(loss5, params5, learning_rate=0.01, momentum=0.9)
    test_prediction5 = lasagne.layers.get_output(network, deterministic=True)
    test_loss5 = lasagne.objectives.squared_error(test_prediction5,target_var2)
    test_loss5 = test_loss5.mean()
    train_fn5 = theano.function([input_var, target_var2], loss5, updates=updates5)
    val_fn5 = theano.function([input_var, target_var2], [test_loss5, test_prediction5])

    # Finally, launch the training loop.
    print("Starting training for 4th auto...")
    # We iterate over epochs:
    valloss = []
    testpred = []
    valpred = []
    patience = 0
    epoch = 0
    its = 0
    while patience<50 and epoch < num_epochs or its >165000:

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(type='train'):
            inputs = batch
            train_err += train_fn5(inputs, inputs)
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        temp = []
        for batch in iterate_minibatches( type='val'):
            inputs = batch
            err, pred = val_fn5(inputs, inputs)
            val_err += err
            val_batches += 1


            if num_epochs-epoch<3 or patience>6:

                if not any(temp):
                    temp = pred.tolist()
                else:
                    temp = temp + pred.tolist()

        valloss.append(val_err/val_batches)
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
                np.savez(parname, *lasagne.layers.get_all_param_values(pool1))
                patience=0

        epoch+=1


    return valloss,bestloss

