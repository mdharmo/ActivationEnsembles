# Author: Mark Harmon
# Various activation functions
# Also, some from Bengio's new paper...
# It appears that unfortunately, p is not solved for in this functions.  It must be elsewhere.

import theano.tensor as T
import numpy as np
def n_relucap(x):
    y = T.switch(x<0,0,x)
    return T.switch(y>1.,1.,y)

def n_relu(x):

    return T.nnet.relu(x)


def n_reluleft(x):
    return T.nnet.relu(-x)

def n_tanh(x):
    #Changed here
    return T.tanh(x)

def n_softrelu(x):
    y = T.nnet.sigm.softplus(x)

    return y

def n_invabs(x):
    #Changed here as well
    return x/(1.0+T.abs_(x))

def n_relu1(x):
    y = T.switch((x - 1.0) < 0, 0, (x - 1.0))

    return y


def n_relu2(x):
    y = T.switch((x - 0.5) < 0, 0, (x - 0.5))

    return y


def n_relu3(x):
    y = T.switch(x < 0., 0., x)

    return y


def n_relu4(x):
    y = T.switch((x + 0.5) < 0, 0, (x + 0.5))

    return y


def n_relu5(x):
    y = T.switch((x + 1.) < 0, 0,(x + 1.))

    return y

def n_sig(x):
    y= T.nnet.sigmoid(x)

    return y

def n_explin(x):
    y = T.switch(x>0,x,T.exp(x)-1.)
    return y


def n_relu1cap(x):
    y = T.switch((x - 0.5) < 0, 0, (x - 0.5))

    return T.switch(y>=0.5,0.5,y)


def n_relu2cap(x):
    y = T.switch((x - 0.25) < 0, 0, (x - 0.25))

    return T.switch(y>=0.5,0.5,y)


def n_relu3cap(x):
    y = T.switch(x < 0, 0, x)

    return T.switch(y>=0.5,0.5,y)


def n_relu4cap(x):
    y = T.switch((x + 0.25) < 0, 0, (x + 0.25))

    return T.switch(y>=0.5,0.5,y)


def n_relu5cap(x):
    y = T.switch((x + 0.5) < 0, 0,(x + 0.5))

    return T.switch(y>=0.5,0.5,y)

def n_threshrelu(x,deterministic = False):
    y = T.switch(x>1,1,0)
    return y*T.switch(x>0,x,0)

def n_threshlin(x,deterministic=False):
    y = T.switch(T.abs_(x)>1,1,0)
    return y*T.switch(x>0,x,0)

def n_lin(x):
    return x
def n_quad(x):
    return T.sqr(x)
def n_three(x):
    y = T.sqr(x)
    return y*x

def clip_sigmoid(x):
    y = T.nnet.sigmoid(x)
    y = T.switch(y<0.00035,0,y)
    return y

def leaky_relu(x):
    return T.nnet.relu(x, 0.01)

def n_leftleaky_relu(x):
    return T.nnet.relu(-x,0.01)