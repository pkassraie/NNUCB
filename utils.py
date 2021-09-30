from neural_tangents import stax
from jax.api import jit
import torch.nn as nn
import numpy as np

def get_kernel(width=100, depth=1, type='NN'):
    '''
    Returns the infinite width kernel of the network.
    :param width: width of the net, same across all layers
    :param depth: number of layers.
    :param type: Can choose from NN, CNN
    :return: jax kernel function
    '''
    if type == 'NN':
        module = stax.Dense(width)
    elif type == 'CNN':
        module = stax.Conv(width, (3, 3), (1, 1), padding='SAME')
    else:
        raise RuntimeError('Kernel is not Defined, yet!')
    layers = []
    for _ in range(depth):
        layers += [module, stax.Relu()]
    layers += [stax.Dense(1)]

    _, _, kernel = stax.serial(*layers)
    return jit(kernel, static_argnums=(2,))


def fit_sublinear(x, y, with_coefs = True):
    coefs = np.polyfit(np.log(x), np.log(y), 1)
    # coefs = np.polyfit(specra, np.log(eigs), 1) # for exponential fit
    poly = np.power(x, coefs[0]) * np.exp(coefs[1])
    if with_coefs:
        return coefs, poly
    return poly

def normalize_init(net):
    '''
    :param net: input network
    :return: network that is normalized wrt xavier initialization
    '''
    layers_list = [module for module in net.modules() if type(module) != nn.Sequential]
    for layer in layers_list:
        if type(layer)== nn.Linear:
            nn.init.xavier_normal_(layer.weight,gain=2)
            layer.bias.data.fill_(0.0)
    return net



