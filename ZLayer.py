# Author: Mark Harmon
# Purpose: Z Project implanted into lasagne structure
import theano
import theano.tensor as T
from lasagne import nonlinearities
import lasagne
import numpy as np
from lasagne.random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "ZLayer"
]



class ZLayer(lasagne.layers.Layer):
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
    def __init__(self, incoming, num_units,axes='auto', z = lasagne.init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,feed=True,n=None,p = 0.,rescale=True,
                 delta = lasagne.init.Constant(1.),eta = lasagne.init.Constant(0.),
                 running_min = lasagne.init.Constant(0.),running_max = lasagne.init.Constant(1.),
                 epsilon=1e-4,**kwargs):

        super(ZLayer, self).__init__(incoming, **kwargs)


        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        self.axes = axes
        self.epsilon = epsilon

        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        # This is for dropout
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale

        # The rest in this definition is for z-layer
        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n+2, n))
        self.n = n
        self.num_units = num_units
        self.feed = feed

        # Lasagne version
        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        shape = [len(nonlinearity)] + shape

        if z is None:
            self.z = None
        else:
            z = lasagne.init.Constant(1/float(len(nonlinearity)))
            self.z = self.add_param(z, (len(nonlinearity),num_units,),name="z",regularlizable=False)

        if delta is None:
            self.delta = None
        else:
            self.delta = self.add_param(delta, shape,name="delta",regularlizable=False)

        if eta is None:
            self.eta = None
        else:
            self.eta = self.add_param(eta, shape, name="eta", regularlizable=False)

        self.running_min = self.add_param(running_min, shape, name="running_min", regularlizable=False)

        self.running_max = self.add_param(running_max, shape, name="running_max", regularlizable=False)


    def get_output_for(self, input,deterministic=False, **kwargs):

        activation = input
        val = T.zeros_like(activation)
        # If probability is zero or if this is the test/validation set
        if deterministic:
            if self.feed:
                for i,non in enumerate(self.nonlinearity):
                    temp = self.nonlinearity[i](activation)
                    temp = self.delta[i]*(temp-T.min(temp,axis=0))/(T.max(temp,axis=0)-T.min(temp,axis=0) + self.epsilon) + self.eta[i]
                    #temp = self.nonlinearity[i](activation)
                    val += self.z[i]*temp
            else:
                for i,non in enumerate(self.nonlinearity):
                    temp = self.nonlinearity[i](activation)
                    temp = self.delta[i]*(temp-T.min(temp,axis=0))/(T.max(temp,axis=0)-T.min(temp,axis=0) + self.epsilon)+self.eta[i]
                    #temp = self.nonlinearity[i](activation)
                    val += self.z[i].dimshuffle(('x',0) + ('x',) * self.n)*temp

        else:
            if self.feed:
                for i,non in enumerate(self.nonlinearity):
                    temp = self.nonlinearity[i](activation)
                    temp = self.delta[i]*(temp-self.running_min[i])/(self.running_max[i]-self.running_min[i] + self.epsilon) + self.eta[i]
                    #temp = self.nonlinearity[i](activation)
                    val += self.z[i]*temp
            else:
                for i,non in enumerate(self.nonlinearity):
                    temp = self.nonlinearity[i](activation)
                    temp = self.delta[i]*(temp-self.running_min[i])/(self.running_max[i]-self.running_min[i] + self.epsilon)+self.eta[i]
                    #temp = self.nonlinearity[i](activation)
                    val += self.z[i].dimshuffle(('x',0) + ('x',) * self.n)*temp


        return val
