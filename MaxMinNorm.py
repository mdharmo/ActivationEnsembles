import theano
import theano.tensor as T
from lasagne import nonlinearities
import lasagne
import numpy as np
from lasagne.random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


__all__ = [
    "MaxMinNorm"
]



class MaxMinNorm(lasagne.layers.Layer):
    """
    lasagne.layers.BatchNormLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs)
    Batch Normalization
    This layer implements batch normalization of its inputs, following [1]_:
    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta
    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.
    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.
    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.
    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.
    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.
    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.
    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.
    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer
    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, incoming,num_units,z = lasagne.init.Constant(0.), axes='auto', epsilon=1e-4, alpha=0.1,
                 eta = lasagne.init.Constant(1.),delta = lasagne.init.Constant(0.),
                 mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1),
                 nonlinearity=[nonlinearities.rectify],feed=True,n=None,**kwargs):
        super(MaxMinNorm, self).__init__(incoming, **kwargs)

        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha


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
        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        shape = [len(nonlinearity)] + shape

        # Lasagne version
        if z is None:
            self.z = None
        else:
            #z = lasagne.init.Constant(1/float(len(nonlinearity)))
            self.z = self.add_param(z, shape,name="z",regularlizable=False)


        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")

        if eta is None:
            self.eta = None
        else:
            self.eta = self.add_param(eta, shape, 'eta',
                                       trainable=True, regularizable=False)
        if delta is None:
            self.delta = None
        else:
            self.delta = self.add_param(delta, shape, 'delta',
                                        trainable=True, regularizable=True)

        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        #shape = [len(self.nonlinearity)] + self.input_shape
        #val = T.zeros_like(input)

        input_mean = T.zeros_like(self.mean)
        input_inv_std = T.zeros_like(self.mean)


        if len(self.nonlinearity)==6:
            act1 = self.nonlinearity[0](input)
            input_mean = T.set_subtensor(input_mean[0],T.min(act1,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[0],(T.max(act1,axis=self.axes)-T.min(act1,axis=self.axes) + self.epsilon))

            act2 = self.nonlinearity[1](input)
            input_mean = T.set_subtensor(input_mean[1],T.min(act2,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[1],(T.max(act2,axis=self.axes)-T.min(act2,axis=self.axes) + self.epsilon))

            act3 = self.nonlinearity[2](input)
            input_mean = T.set_subtensor(input_mean[2],T.min(act3,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[2],(T.max(act3,axis=self.axes)-T.min(act3,axis=self.axes) + self.epsilon))

            act4 = self.nonlinearity[3](input)
            input_mean = T.set_subtensor(input_mean[3],T.min(act4,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[3],(T.max(act4,axis=self.axes)-T.min(act4,axis=self.axes) + self.epsilon))

            act5 = self.nonlinearity[4](input)
            input_mean = T.set_subtensor(input_mean[4],T.min(act5,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[4],(T.max(act5,axis=self.axes)-T.min(act5,axis=self.axes) + self.epsilon))

            act6 = self.nonlinearity[5](input)
            input_mean = T.set_subtensor(input_mean[5],T.min(act6,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[5],(T.max(act6,axis=self.axes)-T.min(act6,axis=self.axes) + self.epsilon))

        elif len(self.nonlinearity)==5:
            act1 = self.nonlinearity[0](input)
            input_mean = T.set_subtensor(input_mean[0],T.min(act1,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[0],(T.max(act1,axis=self.axes)-T.min(act1,axis=self.axes) + self.epsilon))

            act2 = self.nonlinearity[1](input)
            input_mean = T.set_subtensor(input_mean[1],T.min(act2,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[1],(T.max(act2,axis=self.axes)-T.min(act2,axis=self.axes) + self.epsilon))

            act3 = self.nonlinearity[2](input)
            input_mean = T.set_subtensor(input_mean[2],T.min(act3,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[2],(T.max(act3,axis=self.axes)-T.min(act3,axis=self.axes) + self.epsilon))

            act4 = self.nonlinearity[3](input)
            input_mean = T.set_subtensor(input_mean[3],T.min(act4,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[3],(T.max(act4,axis=self.axes)-T.min(act4,axis=self.axes) + self.epsilon))

            act5 = self.nonlinearity[4](input)
            input_mean = T.set_subtensor(input_mean[4],T.min(act5,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[4],(T.max(act5,axis=self.axes)-T.min(act5,axis=self.axes) + self.epsilon))
        else:
            act1 = self.nonlinearity[0](input)
            input_mean = T.set_subtensor(input_mean[0],T.min(act1,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[0],(T.max(act1,axis=self.axes)-T.min(act1,axis=self.axes) + self.epsilon))

            act2 = self.nonlinearity[1](input)
            input_mean = T.set_subtensor(input_mean[1],T.min(act2,axis=self.axes))
            input_inv_std = T.set_subtensor(input_inv_std[1],(T.max(act2,axis=self.axes)-T.min(act2,axis=self.axes) + self.epsilon))


        if use_averages:

            #mean = input_mean
            #inv_std = input_inv_std
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std


        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0*running_mean
            inv_std += 0*running_inv_std

        if len(self.nonlinearity)==6:
            tempz = self.z[0].dimshuffle(pattern)
            tempmean = mean[0].dimshuffle(pattern)
            tempstd = inv_std[0].dimshuffle(pattern)
            tempeta = self.eta[0].dimshuffle(pattern)
            tempdelta = self.delta[0].dimshuffle(pattern)
            val = tempz * ((tempeta*(act1 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[1].dimshuffle(pattern)
            tempmean = mean[1].dimshuffle(pattern)
            tempstd = inv_std[1].dimshuffle(pattern)
            tempeta = self.eta[1].dimshuffle(pattern)
            tempdelta = self.delta[1].dimshuffle(pattern)
            val += tempz * ((tempeta*(act2 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[2].dimshuffle(pattern)
            tempmean = mean[2].dimshuffle(pattern)
            tempstd = inv_std[2].dimshuffle(pattern)
            tempeta = self.eta[2].dimshuffle(pattern)
            tempdelta = self.delta[2].dimshuffle(pattern)
            val += tempz * ((tempeta*(act3 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[3].dimshuffle(pattern)
            tempmean = mean[3].dimshuffle(pattern)
            tempstd = inv_std[3].dimshuffle(pattern)
            tempeta = self.eta[3].dimshuffle(pattern)
            tempdelta = self.delta[3].dimshuffle(pattern)
            val += tempz * ((tempeta*(act4 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[4].dimshuffle(pattern)
            tempmean = mean[4].dimshuffle(pattern)
            tempstd = inv_std[4].dimshuffle(pattern)
            tempeta = self.eta[4].dimshuffle(pattern)
            tempdelta = self.delta[4].dimshuffle(pattern)
            val += tempz * ((tempeta*(act5 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[5].dimshuffle(pattern)
            tempmean = mean[5].dimshuffle(pattern)
            tempstd = inv_std[5].dimshuffle(pattern)
            tempeta = self.eta[5].dimshuffle(pattern)
            tempdelta = self.delta[5].dimshuffle(pattern)
            val += tempz * ((tempeta*(act6 - tempmean) / tempstd) + tempdelta)

        elif len(self.nonlinearity)==5:
            tempz = self.z[0].dimshuffle(pattern)
            tempmean = mean[0].dimshuffle(pattern)
            tempstd = inv_std[0].dimshuffle(pattern)
            tempeta = self.eta[0].dimshuffle(pattern)
            tempdelta = self.delta[0].dimshuffle(pattern)
            val = tempz * ((tempeta*(act1 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[1].dimshuffle(pattern)
            tempmean = mean[1].dimshuffle(pattern)
            tempstd = inv_std[1].dimshuffle(pattern)
            tempeta = self.eta[1].dimshuffle(pattern)
            tempdelta = self.delta[1].dimshuffle(pattern)
            val += tempz * ((tempeta*(act2 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[2].dimshuffle(pattern)
            tempmean = mean[2].dimshuffle(pattern)
            tempstd = inv_std[2].dimshuffle(pattern)
            tempeta = self.eta[2].dimshuffle(pattern)
            tempdelta = self.delta[2].dimshuffle(pattern)
            val += tempz * ((tempeta*(act3 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[3].dimshuffle(pattern)
            tempmean = mean[3].dimshuffle(pattern)
            tempstd = inv_std[3].dimshuffle(pattern)
            tempeta = self.eta[3].dimshuffle(pattern)
            tempdelta = self.delta[3].dimshuffle(pattern)
            val += tempz * ((tempeta*(act4 - tempmean) / tempstd) + tempdelta)

            tempz = self.z[4].dimshuffle(pattern)
            tempmean = mean[4].dimshuffle(pattern)
            tempstd = inv_std[4].dimshuffle(pattern)
            tempeta = self.eta[4].dimshuffle(pattern)
            tempdelta = self.delta[4].dimshuffle(pattern)
            val += tempz * ((tempeta*(act5 - tempmean) / tempstd) + tempdelta)
        else:
            tempz = self.z[0].dimshuffle(pattern)
            tempmean = mean[0].dimshuffle(pattern)
            tempstd = inv_std[0].dimshuffle(pattern)
            tempeta = self.eta[0].dimshuffle(pattern)
            tempdelta = self.delta[0].dimshuffle(pattern)
            val = tempz * ((tempeta*(act1 - tempmean) / tempstd) + tempdelta)
            #val = tempz * (((act1 - tempmean) / tempstd))
            #val += 0*tempeta + 0*tempdelta

            tempz = self.z[1].dimshuffle(pattern)
            tempmean = mean[1].dimshuffle(pattern)
            tempstd = inv_std[1].dimshuffle(pattern)
            tempeta = self.eta[1].dimshuffle(pattern)
            tempdelta = self.delta[1].dimshuffle(pattern)
            val += tempz * ((tempeta*(act2 - tempmean) / tempstd) + tempdelta)
            #val += tempz * (((act2 - tempmean) / tempstd))
            #val +=0*tempeta + 0*tempdelta


        return val



