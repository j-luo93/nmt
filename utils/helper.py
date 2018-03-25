from __future__ import division, print_function

import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.nn import Parameter
import numpy as np

import pprint

NEG_INF = -999999

# shortcuts
tanh = nn.functional.tanh
relu = nn.functional.relu
leaky_relu = nn.functional.leaky_relu
sigmoid = nn.functional.sigmoid

def sync():
    torch.cuda.synchronize()

def get_tensor(tensor):
    if 'USE_CUDA' in os.environ:
        tensor = tensor.cuda()
    return tensor

def get_variable(tensor, **kwargs):
    v = V(tensor, **kwargs)
    if os.environ.get('USE_CUDA', None) == '1' and not v.is_cuda:
        v = v.cuda()
    return v

def get_zeros(sizes, training=True, tensor=False):
    volatile = not training
    if os.environ.get('USE_CUDA', None) == '1':
        h = torch.cuda.FloatTensor(*sizes).fill_(0.0)
    else:
        h = torch.zeros(*sizes)
    if not tensor:
        h = V(h, volatile=volatile)
    return h

def get_values(sizes, value): 
    h = get_zeros(sizes, tensor=True)
    h.fill_(value)
    return V(h)
    
def get_type(x):
    assert isinstance(x, V) or torch.is_tensor(x) or isinstance(x, LSTMState)
    if isinstance(x, V):
        x = x.data
    elif isinstance(x, LSTMState):
        x = x.states[0][0].data
    type_ = x.type()
    types = ['Float', 'Long', 'Byte']
    for t in types:
        if t in type_:
            return t.lower()
    raise NotImplementedError
    
def expand(x, dim, ns):
    sizes = list(x.size())
    sizes.insert(dim, ns)
    res = x.unsqueeze(dim=dim).expand(*sizes).contiguous()
    sizes.pop(dim)
    if dim == 0:
        sizes[0] *= ns
    else:
        sizes[dim - 1] *= ns
    return res.view(*sizes)

def where(mask, a, b):
    res = b.clone()
    res[mask] = a[mask]
    return res
