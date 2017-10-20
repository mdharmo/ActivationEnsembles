# Author: Mark Harmon
# Various activation functions
# Also, some from Bengio's new paper...
# It appears that unfortunately, p is not solved for in this functions.  It must be elsewhere.

import numpy as np

def n_relu(x):
    y = np.maximum(0,x)
    return y

def n_reluleft(x):
    y = np.maximum(0,-x)
    return y

def n_tanh(x):
    y = np.tanh(x)
    return y

def n_softrelu(x):
    y = np.log(1.0+np.exp(x))
    return y

def n_sig(x):
    y = 1.0/(1.0 + np.exp(-1.*x))
    return y

def n_invabs(x):
    #Changed here as well
    y = x/(1.0+np.abs(x))
    return y

def n_relu1(x):
    y = np.maximum(0,x-1.0)
    return y


def n_relu2(x):
    y = np.maximum(0,x-0.5)
    return y


def n_relu3(x):
    y = np.maximum(0,x)
    return y


def n_relu4(x):
    y = np.maximum(0,x+0.5)
    return y


def n_relu5(x):
    y = np.maximum(0,x+1.0)
    return y


def n_explin(x):
    y = np.minimum(x,np.exp(x)-1)
    return y

