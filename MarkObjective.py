import theano.tensor as T
import numpy as np
from lasagne.utils import as_theano_expression

def align_targets(predictions, targets):
    """Helper function turning a target 1D vector into a column if needed.
    This way, combining a network of a single output unit with a target vector
    works as expected by most users, not broadcasting outputs against targets.
    Parameters
    ----------
    predictions : Theano tensor
        Expression for the predictions of a neural network.
    targets : Theano tensor
        Expression or variable for corresponding targets.
    Returns
    -------
    predictions : Theano tensor
        The predictions unchanged.
    targets : Theano tensor
        If `predictions` is a column vector and `targets` is a 1D vector,
        returns `targets` turned into a column vector. Otherwise, returns
        `targets` unchanged.
    """
    if (getattr(predictions, 'broadcastable', None) == (False, True) and
            getattr(targets, 'ndim', None) == 1):
        targets = as_theano_expression(targets).dimshuffle(0, 'x')
    return predictions, targets

def categorical_crossentropy(predictions, targets):

    """Computes the categorical cross-entropy between predictions and targets.
    .. math:: L_i = - \\sum_j{t_{i,j} \\log(p_{i,j})}
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either targets in [0, 1] matching the layout of `predictions`, or
        a vector of int giving the correct class index per data point.
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical cross-entropy.
    Notes
    -----
    This is the loss function of choice for multi-class classification
    problems and softmax output units. For hard targets, i.e., targets
    that assign all of the probability to a single class per data point,
    providing a vector of int for the targets is usually slightly more
    efficient than providing a matrix with a single 1.0 per row.
    """
    _EPSILON = np.finfo(np.float32).eps
    predictions = T.clip(predictions, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.categorical_crossentropy(predictions, targets)

def binary_crossentropy(predictions, targets):
    """Computes the binary cross-entropy between predictions and targets.
    .. math:: L = -t \\log(p) - (1 - t) \\log(1 - p)
    Parameters
    ----------
    predictions : Theano tensor
        Predictions in (0, 1), such as sigmoidal output of a neural network.
    targets : Theano tensor
        Targets in [0, 1], such as ground truth labels.
    Returns
    -------
    Theano tensor
        An expression for the element-wise binary cross-entropy.
    Notes
    -----
    This is the loss function of choice for binary classification problems
    and sigmoid output units.
    """
    predictions, targets = align_targets(predictions, targets)
    _EPSILON = np.finfo(np.float32).eps
    predictions = T.clip(predictions, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.binary_crossentropy(predictions, targets)