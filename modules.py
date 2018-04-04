from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils.helper import get_zeros, get_variable, where, get_values
from utils.LSTMState import LSTMState
from models_utils import NEG_INF
from log_uniform.log_uniform import LogUniformSampler

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
        
class MoEDecoder(nn.Module):
    
    def __init__(self, input_size, n_experts, tgt_vocab_size, 
                 sampled_softmax=False, 
                 n_samples=None,
                 gumble=False,
                 straight_through=False):
        super(MoEDecoder, self).__init__()
        
        self.n_experts = n_experts
        self.tgt_vocab_size = tgt_vocab_size
        self.sampled_softmax = sampled_softmax
        self.n_samples = n_samples
        self.gumble = gumble
        self.straight_through = straight_through
        # gating
        self.Wg = nn.Parameter(torch.Tensor(input_size, n_experts)) 
        # experts
        if self.sampled_softmax:
            self.projection = SampledSoftmax(tgt_vocab_size, n_samples, input_size, shared_by_experts=n_experts)
        else:
            self.projection = nn.Linear(input_size, n_experts * tgt_vocab_size)
        if self.gumble:
            self.gumble_softmax = GumbleSoftmax(straight_through=self.straight_through)
        
    def forward(self, input_, labels=None):
        bs = input_.size(0)
        expert_logits = input_.mm(self.Wg)
        if self.gumble:
            expert_probs = self.gumble_softmax(expert_logits)
            import ipdb; ipdb.set_trace()
        else:
            expert_probs = nn.functional.log_softmax(expert_logits, dim=1).exp()
        if self.sampled_softmax:
            all_logits = self.projection(input_, labels)
        else:
            all_logits = self.projection(input_).view(bs, self.n_experts, self.tgt_vocab_size)
        return expert_probs, all_logits

'''
Modified from https://github.com/rdspring1/PyTorch_GBW_LM
'''
class SampledSoftmax(nn.Module):
    
    def __init__(self, ntokens, nsampled, nhid, shared_by_experts=0): 
        ''' If shared_by_experts > 0, then reuse the same sample_id for all experts'''
        
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled
        self.shared_by_experts = shared_by_experts

        self.sampler = LogUniformSampler(self.ntokens)
        
        if self.shared_by_experts > 0:
            self.params = nn.Linear(nhid, ntokens * self.shared_by_experts)
        else:
            self.params = nn.Linear(nhid, ntokens)

    def forward(self, inputs, labels=None):
        if self.training:
            assert labels is not None
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):
        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = get_variable(torch.LongTensor(sample_ids))
        true_freq = get_variable(torch.FloatTensor(true_freq))
        sample_freq = get_variable(torch.FloatTensor(sample_freq))

        if self.shared_by_experts > 0:
            true_weights = self.params.weight.view(self.ntokens, self.shared_by_experts, -1)[labels]
            true_bias = self.params.bias.view(self.ntokens, self.shared_by_experts)[labels]
            
            sample_weights = self.params.weight.view(self.ntokens, self.shared_by_experts, -1)[sample_ids]
            sample_bias = self.params.bias.view(self.ntokens, self.shared_by_experts)[sample_ids]

            true_logits = (inputs.unsqueeze(dim=1) * true_weights).sum(dim=2) + true_bias # bs x ne
            sample_logits = inputs.mm(sample_weights.view(self.nsampled * self.shared_by_experts, -1).t()).view(batch_size, self.nsampled, self.shared_by_experts) + sample_bias # bs x ns x ne
        else:
            # gather true labels - weights and frequencies
            sample_weights = self.params.weight[sample_ids, :]
            sample_bias = self.params.bias[sample_ids]
            
            # gather sample ids - weights and frequencies
            true_weights = self.params.weight[labels, :]
            true_bias = self.params.bias[labels]

            # calculate logits
            true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
            sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        if self.shared_by_experts > 0:
            true_logits = true_logits.sub(torch.log(true_freq).unsqueeze(dim=1))
            sample_logits = sample_logits.sub(torch.log(sample_freq).unsqueeze(dim=1))
        else:
            # perform correction
            true_logits = true_logits.sub(torch.log(true_freq))
            sample_logits = sample_logits.sub(torch.log(sample_freq))
            
        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)

        return logits

    def full(self, inputs):
        if self.shared_by_experts > 0: 
            return self.params(inputs).view(-1, self.ntokens, self.shared_by_experts)
        else:
            return self.params(inputs)

'''
Modified from https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530
'''
class GumbleSoftmax(nn.Module):
    
    def __init__(self, straight_through=False):
        super(GumbleSoftmax, self).__init__()
        self.temperature = 1.0
        self.straight_through = straight_through
    
    def anneal(self):
        raise NotImplementedError()
    
    def sample(self, input_, eps=1e-20):
        noise = torch.rand(input_.size())
        noise = -(-(noise + eps).log() + eps).log()
        return get_variable(noise)
        
    def forward(self, input_):
        noise = self.sample(input_)
        x = (input_ + noise) / self.temperature
        x = nn.functional.log_softmax(x, dim=1).exp() # NOTE use exp() since it's used for expert gating
        if self.straight_through:
            x_hard_val, _ = x.max(dim=1, keepdim=True) # bs x 1
            x_hard = (x_hard_val == x).float()
            return (x_hard - x).detach() + x
        else:
            return x