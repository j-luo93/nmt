from __future__ import print_function, division

import torch
import torch.nn as nn
import math
import numbers
from operator import mul

from utils.helper import get_zeros, get_variable, where, get_values, NEG_INF
from utils.LSTMState import LSTMState
from models_utils import NEG_INF

class MaskedNLLLoss(nn.Module):
    
    def forward(self, logits, targets, mask=None, dim=-1):
        log_probs = nn.functional.log_softmax(logits, dim=dim)
        res = log_probs.gather(dim, targets.unsqueeze(dim=2)).squeeze(dim=2)
        if mask is not None:
            res = res * mask
        return -res

class MultiLayerRNNCell(nn.Module):
    
    def __init__(self, num_layers, input_size, hidden_size, module='LSTM', dropout=0.0):
        super(MultiLayerRNNCell, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.module = module
        self.drop = nn.Dropout(dropout)
        
        assert module in ['LSTM', 'GRU']
        assert module == 'LSTM'
        mod = nn.LSTMCell if module == 'LSTM' else nn.GRUCell
        cells = [mod(input_size, hidden_size)] + \
                [mod(hidden_size, hidden_size) for _ in xrange(self.num_layers - 1)]
        self.cells = nn.ModuleList(cells)
    
    def init_state(self, bs, encoding=None):
        states = list()
        for _ in xrange(self.num_layers):
            if encoding is None:
                state = get_zeros([bs, self.hidden_size], training=self.training)
                if self.module == 'LSTM':
                    c = get_zeros([bs, self.hidden_size], training=self.training)
                    state = (state, c)
            else:
                assert encoding[0].size(0) == 2
                state = (encoding[0].mean(dim=0), encoding[1].mean(dim=0))
            states.append(state)
        return LSTMState(states)
        
    def forward(self, input_, states):
        assert len(states) == self.num_layers
        
        new_states = list()
        for i in xrange(self.num_layers):
            new_state = self.cells[i](input_, states.get(i))
            new_states.append(new_state)
            input_ = new_state[0] if self.module == 'LSTM' else new_state
            input_ = self.drop(input_)
        return LSTMState(new_states)

class GlobalAttention(nn.Module):
    
    def __init__(self, cell_dim, dropout, hidden_size=None):
        super(GlobalAttention, self).__init__()
        self.cell_dim = cell_dim
        hidden_size = cell_dim if hidden_size is None else hidden_size

        inp_dim = 2 * cell_dim # bidirectional

        self.Wa = nn.Parameter(torch.Tensor(inp_dim, hidden_size))
        self.hidden = nn.Linear(2 * cell_dim + hidden_size, cell_dim)
        self.drop = nn.Dropout(dropout)

    def precompute(self, h_s):
        l = h_s.size(0)
        bs = h_s.size(1)
        return h_s.view(l * bs, -1).mm(self.Wa).view(l, bs, -1)
        
    def forward(self, Wh_s, h_t, mask_src, h_s):
        sl, bs, d = h_s.size()
        scores = (h_t.expand_as(Wh_s) * Wh_s).sum(dim=2) # sl x bs
        # softmax   
        alignment = nn.functional.log_softmax(scores.t()).t().exp() * mask_src # sl x bs
        alignment = alignment / (alignment.sum(dim=0) + 1e-8)
        
        context = (alignment.unsqueeze(dim=2).expand_as(h_s) * h_s).sum(dim=0)#.squeeze(dim=0) # bs x n_src
        cat = torch.cat([context, h_t], 1)
        h_tilde = nn.functional.tanh(self.hidden(self.drop(cat)))
        return h_tilde, alignment.t().contiguous() # NOTE alignment now bs x sl, used for loss module
        
class PositionalEncoding(nn.Module):
    
    def __init__(self, cell_dim, denominator=10000):
        super(PositionalEncoding, self).__init__()
        assert cell_dim % 2 == 0, 'cell dim must be even'
        self.cell_dim = cell_dim
        
        pow = get_variable(torch.arange(1, self.cell_dim // 2 + 1) * 2 / self.cell_dim) # NOTE starting with 1
        self.base = torch.pow(get_values(pow.size(), denominator), pow)
    
    def forward(self, position): 
        x = (position.unsqueeze(dim=2) + 1).float() / self.base # NOTE starting with 1
        return torch.cat([torch.sin(x), torch.cos(x)], 2)

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, cell_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.cell_dim = cell_dim
        
    def forward(self, query, key, value, use_mask=False): # corresponding to Q, K, and V in the original paper
        bs, _, d = value.size()
        ql = query.size(1)
        kl = key.size(1)
        mm = query.matmul(key.transpose(1, 2)) # bs x ql x kl
        if use_mask:
            max_lengths = get_variable(torch.arange(0, ql).long().view(1, ql, 1))
            indices = get_variable(torch.arange(0, kl).long().view(1, 1, kl).expand(bs, ql, kl))
            mask = (indices <= max_lengths).float()
            mm = mm + mask * NEG_INF
        w = nn.functional.log_softmax(mm / math.sqrt(self.cell_dim), dim=2).exp()
        res = (w.contiguous().view(bs, ql, kl, 1) * value.contiguous().view(bs, 1, kl, d)).sum(dim=2) # bs x ql x d
        return res
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, cell_dim, query_size, key_size, value_size):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.cell_dim = cell_dim
        
        self.Wq = nn.Linear(query_size, cell_dim * n_heads)
        self.Wk = nn.Linear(key_size, cell_dim * n_heads)
        self.Wv = nn.Linear(value_size, cell_dim * n_heads)
        self.Wo = nn.Linear(cell_dim * n_heads, cell_dim * n_heads)
        
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.cell_dim)
        
    def forward(self, query, key, value, use_mask=False):
        qs = self.Wq(query).chunk(self.n_heads, dim=2)
        ks = self.Wk(key).chunk(self.n_heads, dim=2)
        vs = self.Wv(value).chunk(self.n_heads, dim=2)
        
        heads = list()
        for q, k, v in zip(qs, ks, vs):
            heads.append(self.scaled_dot_product_attention(q, k, v, use_mask=use_mask)) 
        res = self.Wo(torch.cat(heads, dim=2))
        return res

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), 
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, output_size))
    
    def forward(self, input_):
        return self.network(input_)
        
class EncoderLayer(nn.Module):
    
    def __init__(self, input_size, n_heads):
        super(EncoderLayer, self).__init__()
        assert input_size % n_heads == 0
        self.self_attention = MultiHeadAttention(n_heads, input_size // n_heads, input_size, input_size, input_size)
        self.layer_norm_self_attention = LayerNorm(input_size)
        self.feed_forward = MLP(input_size, input_size * 4, input_size)
        self.layer_norm_MLP = LayerNorm(input_size)
        
    def forward(self, input_):
        h1 = self.self_attention(input_, input_, input_)
        h2 = self.layer_norm_self_attention(h1 + input_)
        h3 = self.feed_forward(h2)
        h4 = self.layer_norm_MLP(h3 + h2)
        return h4
        
class DecoderLayer(nn.Module):
    
    def __init__(self, input_size, n_heads):
        super(DecoderLayer, self).__init__()
        assert input_size % n_heads == 0
        self.self_attention = MultiHeadAttention(n_heads, input_size // n_heads, input_size, input_size, input_size)
        self.layer_norm_self_attention = LayerNorm(input_size)
        self.attention = MultiHeadAttention(n_heads, input_size // n_heads, input_size, input_size, input_size)
        self.layer_norm_attention = LayerNorm(input_size)
        self.feed_forward = MLP(input_size, input_size * 4, input_size)
        self.layer_norm_MLP = LayerNorm(input_size)
        
    def forward(self, input_, encoder_states):
        h1 = self.self_attention(input_, input_, input_, use_mask=self.training)
        h2 = self.layer_norm_self_attention(h1 + input_)
        h3 = self.attention(h2, encoder_states, encoder_states)
        h4 = self.layer_norm_attention(h3 + h2)
        h5 = self.feed_forward(h4)
        h6 = self.layer_norm_MLP(h5 + h4)
        return h6

class EncoderStack(nn.Module):
        
    def __init__(self, input_size, n_heads, num_layers):
        super(EncoderStack, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(input_size, n_heads) for _ in xrange(num_layers)])
    
    def forward(self, input_enc):
        res = list()
        encoder_states = input_enc
        for layer in self.layers:
            encoder_states = layer(encoder_states)
        return encoder_states

class DecoderStack(nn.Module):
    
    def __init__(self, input_size, n_heads, num_layers):
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(input_size, n_heads) for _ in xrange(num_layers)])
        
    def forward(self, input_dec, encoder_states):
        res = list()
        decoder_states = input_dec
        for layer in self.layers:
            decoder_states = layer(decoder_states, encoder_states)
        return decoder_states

class LayerNorm(nn.Module):

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

# '''
# Copied from pytorch github
# '''
# def batch_norm(input, running_mean, running_var, weight=None, bias=None,
#                training=False, momentum=0.1, eps=1e-5):
#     if training:
#         size = list(input.size())
#         if reduce(mul, size[2:], size[0]) == 1:
#             raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
#     f = torch._C._functions.BatchNorm(running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled)
#     return f(input, weight, bias)
# 
# def layer_norm(input, normalized_shape, running_mean=None, running_var=None,
#                weight=None, bias=None, use_input_stats=True,
#                momentum=0.1, eps=1e-5):
#     if not use_input_stats and (running_mean is None or running_var is None):
#         raise ValueError('Expected running_mean and running_var to be not None when use_input_stats=False')
# 
#     if weight is not None and weight.size() != normalized_shape:
#         raise ValueError('Expected weight to be of same shape as '
#                          'normalized_shape, but got {} weight and '
#                          'normalized_shape={}'.format(weight.size(), normalized_shape))
# 
#     if bias is not None and bias.size() != normalized_shape:
#         raise ValueError('Expected bias to be of same shape as '
#                          'normalized_shape, but got {} bias and '
#                          'normalized_shape={}'.format(bias.size(), normalized_shape))
# 
#     normalized_ndim = len(normalized_shape)
#     input_shape = input.size()
# 
#     if input_shape[-normalized_ndim:] != torch.Size(normalized_shape):
#         raise ValueError('Expected input with shape [*, {}], but got {} input'
#                          .format(', '.join(normalized_shape), list(input_shape)))
# 
#     n = reduce(mul, input_shape[:-normalized_ndim], 1)
# 
#     # Repeat stored stats if necessary
#     if running_mean is not None:
#         running_mean_orig = running_mean
#         running_mean = running_mean_orig.repeat(n)
#     if running_var is not None:
#         running_var_orig = running_var
#         running_var = running_var_orig.repeat(n)
# 
#     # Apply layer norm
#     input_reshaped = input.contiguous().view(1, n, -1)
# 
#     import ipdb; ipdb.set_trace()
#     out = nn.functional.batch_norm(
#         input_reshaped, running_mean, running_var, None, None,
#         use_input_stats, momentum, eps)
# 
#     # Copy back
#     if running_mean is not None:
#         running_mean_orig.fill_(running_mean.mean())
#     if running_var is not None:
#         running_var_orig.fill_(running_var.mean())
# 
#     out = out.view(*input_shape)
# 
#     if weight is not None and bias is not None:
#         return torch.addcmul(bias, 1, out, weight)
#     elif weight is not None:
#         return torch.mul(out, weight)
#     elif bias is not None:
#         return torch.add(out, bias)
#     else:
#         return out
# 
# class LayerNorm(nn.Module):
# 
#     def __init__(self, normalized_shape, eps=1e-5, momentum=0.1,
#                  elementwise_affine=True, track_running_stats=False):
#         super(LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         self.normalized_shape = torch.Size(normalized_shape)
#         self.eps = eps
#         self.momentum = momentum
#         self.elementwise_affine = elementwise_affine
#         self.track_running_stats = track_running_stats
#         if self.elementwise_affine:
#             self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
#             self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)
#         if self.track_running_stats:
#             self.register_buffer('running_mean', torch.zeros(1))
#             self.register_buffer('running_var', torch.ones(1))
#         else:
#             self.register_parameter('running_mean', None)
#             self.register_parameter('running_var', None)
#         self.reset_parameters()
# 
#     def reset_parameters(self):
#         if self.track_running_stats:
#             self.running_mean.zero_()
#             self.running_var.fill_(1)
#         if self.elementwise_affine:
#             self.weight.data.fill_(1)
#             self.bias.data.zero_()
# 
#     def forward(self, input):
#         return layer_norm(
#             input, self.normalized_shape, self.running_mean, self.running_var,
#             self.weight, self.bias, self.training or not self.track_running_stats,
#             self.momentum, self.eps)
# 
#     def __repr__(self):
#         return ('{name}({normalized_shape}, eps={eps}, momentum={momentum},'
#                 ' elementwise_affine={elementwise_affine},'
#                 ' track_running_stats={track_running_stats})'
#                 .format(name=self.__class__.__name__, **self.__dict__))
# 
