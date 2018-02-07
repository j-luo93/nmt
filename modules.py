from __future__ import print_function

import torch
import torch.nn as nn
import math

from utils.helper import get_zeros, get_variable, where, get_values
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
