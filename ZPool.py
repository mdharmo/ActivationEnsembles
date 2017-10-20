import theano.tensor as T
import lasagne
from theano.tensor.signal.pool import pool_2d
import numpy as np

__all__ = [
    "MaxPool2DLayer"
]


def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """
    Compute the output length of a pooling operator
    along a single dimension.
    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.
    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.
    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).
    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length

class Pool2DLayer(lasagne.layers.Layer):
    """
    2D pooling layer
    Performs 2D mean or max-pooling over the two trailing axes
    of a 4D input tensor.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.
    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.
    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.
    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    See Also
    --------
    MaxPool2DLayer : Shortcut for max pooling layer.
    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.
    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, num_filters,stride=None, pad=(0, 0),
                 ignore_border=True, mode='max',z = lasagne.init.Constant(0.),n=None, **kwargs):
        super(Pool2DLayer, self).__init__(incoming, **kwargs)


        if n is None:
            n = len(self.input_shape) - 2
        elif n != len(self.input_shape) - 2:
            raise ValueError("Tried to create a %dD convolution layer with "
                             "input shape %r. Expected %d input dimensions "
                             "(batchsize, channels, %d spatial dimensions)." %
                             (n, self.input_shape, n+2, n))
        self.n = n

        self.pool_size = lasagne.utils.as_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = lasagne.utils.as_tuple(stride, 2)

        self.pad = lasagne.utils.as_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode

        if z is None:
            self.z = None
        else:
            z = lasagne.init.Constant(1 / float(len(mode)))
            self.z = self.add_param(z, (len(mode), num_filters,), name="z", regularlizable=False)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):


        for i in range(len(self.mode)):
            pooled = pool_2d(input,
                             ds=self.pool_size,
                             st=self.stride,
                             ignore_border=self.ignore_border,
                             padding=self.pad,
                             mode=self.mode[i],
                             )

            if i ==0:
                val = T.zeros_like(pooled)

            val+= self.z[i].dimshuffle(('x',0) + ('x',) * self.n)*pooled

        return val



class ZPool2DLayer(Pool2DLayer):
    """
    2D max-pooling layer
    Performs 2D max-pooling over the two trailing axes of a 4D input tensor.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.
    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.
    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.
    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.
    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size,num_filters, stride=None, pad=(0, 0),
                 ignore_border=True, **kwargs):
        super(ZPool2DLayer, self).__init__(incoming,
                                             pool_size,
                                             num_filters,
                                             stride,
                                             pad,
                                             ignore_border,
                                             mode=['max','sum'],
                                             **kwargs)

