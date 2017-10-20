# Author: Mark Harmon
# Purpose: Z Project implanted into lasagne structure
import theano
import theano.tensor as T
from lasagne import nonlinearities
import lasagne
from collections import OrderedDict
from lasagne.updates import get_or_compute_grads
import numpy as np
from lasagne.updates import apply_nesterov_momentum,adadelta
from lasagne.updates import rmsprop
from lasagne.updates import nesterov_momentum
__all__ = [
    "ZDenseLayer",
    "Zupdate"
]


class ZDenseLayer(lasagne.layers.Layer):
    """
    lasagne.layers.DenseLayer(incoming, num_units,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, **kwargs)
    A fully connected layer.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    num_units : int
        The number of units of the layer
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_units)``.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)``.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    Examples
    --------
    from lasagne.layers import InputLayer, DenseLayer
    l_in = InputLayer((100, 20))
    l1 = DenseLayer(l_in, num_units=50)
    Notes
    -----
    If the input to this layer has more than two axes, it will flatten the
    trailing axes. This is useful for when a dense layer follows a
    convolutional layer, for example. It is not necessary to insert a
    :class:`FlattenLayer` in this case.
    """
    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), omega = lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, **kwargs):

        super(ZDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

        # Lasagne version
        if omega is None:
            self.omega = None
        else:
            omega = lasagne.init.Constant(1/float(len(nonlinearity)))
            self.omega = self.add_param(omega, (len(nonlinearity),num_units),name="omega",regularlizable=False)


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        val = T.zeros_like(activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)

        for i,non in enumerate(self.nonlinearity):
            val += self.omega[i,:]*self.nonlinearity[i](activation)

        return val



def ZProject(res,act_num,units,):
    # gradient step

    # do the projection
    div = act_num*np.ones((units,),dtype=theano.config.floatX)
    mask = np.ones((act_num, units), dtype=theano.config.floatX)

    for i in range(act_num):
        res = (1.0 - T.sum(res, axis=0)) /div  + res
        res = res * mask
        # Get masking variable first.
        mintemp = T.min(res, axis=0)
        minmask = T.eq(mintemp, res)
        ltmask = T.lt(res, 0.0)
        tempmask = minmask * ltmask
        mask = mask - tempmask
        mask = T.clip(mask, 0.0, 1.0)

        div = div - T.sum(tempmask, axis=0, dtype=theano.config.floatX)

    return res




def Zupdate(loss_or_grads, params, act_num,units,learning_rate,momentum=0):

    #updates = lasagne.updates.sgd(loss_or_grads, params, learning_rate=0.001)
    #updates = rmsprop(loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6)
    updates = adadelta(loss_or_grads, params)
    #updates = lasagne.updates.momentum(loss_or_grads, params, learning_rate=learning_rate, momentum=0.9)
    #updates = nesterov_momentum(loss_or_grads, params, learning_rate=0.01, momentum=0.9)
    count =0
    for param,i in zip(params,range(len(params))):
        temp = str(param)
        if temp[0] == 'z':

            #Apply my project algorithm
            updates[param] = ZProject(updates[param], act_num[count], units[count])
            count+=1


    return updates
