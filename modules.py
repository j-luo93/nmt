from __future__ import print_function, division

import torch
import torch.nn as nn
import math

from utils.helper import get_zeros, get_variable, where, get_values, NEG_INF
from utils.LSTMState import LSTMState
from models_utils import NEG_INF

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
        
        pow = torch.arange(1, self.cell_dim // 2 + 1).long() # NOTE starting with 1
        self.base = get_variable(torch.pow(denominator, pow))
    
    def forward(self, position, batch_size): # assuming position is just an integer
        x = (position + 1) / self.base # NOTE starting with 1
        return torch.cat([torch.sin(x), torch.cos(x)], 0).view(1, self.cell_dim).expand(batch_size, self.cell_dim)

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, cell_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.cell_dim = cell_dim
        
    def forward(self, query, key, value, mask=False): # corresponding to Q, K, and V in the original paper
        import ipdb; ipdb.set_trace()
        bs, sl, d = value.size()
        mm = query.matmul(key)
        if mask:
            import ipdb; ipdb.set_trace()
            # TODO
        w = nn.functional.log_softmax(mm / math.sqrt(self.cell_dim), dim=2).exp()
        res = (w.view(bs, sl, sl, 1) * value.view(bs, 1, sl, d)).sum(dim=2) # bs x sl x d
        return res
        
class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, cell_dim, query_size, key_size, value_size):
        super(MultiLayerRNNCell, self).__init__()
        self.n_heads = n_heads
        self.cell_dim = cell_dim
        
        self.Wq = nn.Linear(query_size, cell_dim * n_heads)
        self.Wk = nn.Linear(key_size, cell_dim * n_heads)
        self.Wv = nn.Linear(value_size, cell_dim * n_heads)
        self.Wo = nn.Linear(cell_dim * n_heads, cell_dim * n_heads)
        
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.cell_dim)
        
    def forward(self, query, key, value, mask=False):
        import ipdb; ipdb.set_trace()
        qs = self.Wq(query).chunk(self.n_heads, dim=2)
        ks = self.Wk(key).chunk(self.n_heads, dim=2)
        vs = self.Wv(value).chunk(self.n_heads, dim=2)
        
        heads = list()
        for q, k, v in zip(qs, ks, vs):
            heads.append(self.scaled_dot_product_attention(q, k, v, mask=mask))
        res = self.Wo(torch.cat(heads, dim=2))
        return res

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), 
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, output_size))
    
    def forward(self, input_):
        return self.network(MLP)
        
def EncoderLayer(nn.Module):
    
    def __init__(self, input_size, n_heads):
        super(EncoderLayer, self).__init__()
        assert input_size % n_heads == 0
        self.self_attention = MultiHeadAttention(n_heads, input_size // n_heads, input_size, input_size, input_size)
        self.layer_norm_self_attention = nn.LayerNorm(input_size)
        self.feed_forward = MLP(input_size, input_size * 4)
        self.layer_norm_MLP = nn.LayerNorm(input_size)
        
    def forward(self, input_):
        h1 = self.self_attention(input_, input_, input_)
        h2 = self.layer_norm_self_attention(h1 + input_)
        h3 = self.feed_forward(h2)
        h4 = self.layer_norm_MLP(h3 + h2)
        return h4
        
def DecoderLayer(nn.Module):
    
    def __init__(self, input_size, n_heads):
        super(DecoderLayer, self).__init__()
        assert input_size % n_heads == 0
        self.self_attention = MultiHeadAttention(n_heads, input_size // n_heads, input_size, input_size, input_size)
        self.layer_norm_self_attention = nn.LayerNorm(input_size)
        self.attention = MultiHeadAttention(n_heads, input_size // n_heads, input_size, input_size, input_size)
        self.layer_norm_attention = nn.LayerNorm(input_size)
        self.feed_forward = MLP(input_size, input_size * 4)
        self.layer_norm_MLP = nn.LayerNorm(input_size)
        
    def forward(self, input_, encoder_states):
        if self.training:
            h1 = self.self_attention(input_, input_, input_, mask=True)
            h2 = self.layer_norm_self_attention(h1 + input_)
            h3 = self.attention(h2, encoder_states, encoder_states)
            h4 = self.layer_norm_attention(h3 + h2)
            h5 = self.feed_forward(h4)
            h6 = self.layer_norm_MLP(h5 + h4)
        else:
            # TODO 
        return h6

# since the depth for encoder and decode should be the same, we model it as a joint layer.    
def JointLayer(nn.Module):
    
    def __init__(self, input_size, n_heads):
        super(JointLayer, self).__init__()
        self.encoder_layer = EncoderLayer(input_size, n_heads)
        self.decoder_layer = DecoderLayer(input_size, n_heads)
    
    def forward(self, input_enc, input_dec):
        encoder_states = self.encoder_layer(input_enc)
        decoder_states = self.decoder_layer(input_dec, encoder_states)
        return encoder_states, decoder_states