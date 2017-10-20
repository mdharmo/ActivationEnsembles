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
__all__ = [
    "Zupdate"
]



def ZProject(res,act_num,units,):
    # gradient step

    #res = param - learning_rate * grad

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




def Zupdate(loss_or_grads, params, act_num,units,learning_rate=1.0, rho=0.95, epsilon=1e-6,alpha = 0.1):



    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    count =0
    one = T.constant(1)
    for param, grad in zip(params, grads):

        temp = str(param)

        if temp[0]!='r':
            value = param.get_value(borrow=True)
            # accu: accumulate gradient magnitudes
            accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                 broadcastable=param.broadcastable)
            # delta_accu: accumulate update magnitudes (recursively!)
            delta_accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                       broadcastable=param.broadcastable)

            # update accu (as in rmsprop)
            accu_new = rho * accu + (one - rho) * grad ** 2
            updates[accu] = accu_new

            # compute parameter update, using the 'old' delta_accu
            update = (grad * T.sqrt(delta_accu + epsilon) /
                      T.sqrt(accu_new + epsilon))
            updates[param] = param - learning_rate * update

            # update delta_accu (as accu, but accumulating updates)
            delta_accu_new = rho * delta_accu + (one - rho) * update ** 2
            updates[delta_accu] = delta_accu_new


            if temp[0] == 'z':

                #Apply my project algorithm
                updates[param] = ZProject(updates[param], act_num[count], units[count])
                count+=1

        else:
            updates[param] = (1-alpha)*updates[param] +

    return updates
