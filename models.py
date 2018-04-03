from __future__ import division, print_function

import cPickle
import numpy as np
import torch
import torch.nn as nn

from utils.helper import get_variable, get_zeros, expand, get_tensor, where, get_values
from utils.vocab import PAD_ID, SOS_ID, EOS_ID
from utils.LSTMState import LSTMState
from utils.variable_dict import VariableDict as VD
from modules import MultiLayerRNNCell, GlobalAttention, MoEDecoder, SampledSoftmax
from models_utils import Record, NEG_INF, cat, AttentionHelper, Beam

class BaseModel(nn.Module):
        
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def encode(self, batch):
        volatile = not self.training
        rsl = batch.real_src_length.cpu().numpy()
        input_enc = get_variable(batch.input_enc, volatile=volatile) 
        mask_src = (input_enc != PAD_ID).float()
        sl, bs = input_enc.size()
        inp_enc = self.src_emb(input_enc)
        #inp_enc = self.drop(inp_enc) # NOTE dropout
        inp_packed = nn.utils.rnn.pack_padded_sequence(inp_enc, rsl)
        h = get_zeros([2 * self.num_layers, bs, self.cell_dim], training=self.training) # NOTE bidirectional, therefore 2
        c = get_zeros([2 * self.num_layers, bs, self.cell_dim], training=self.training)
        annotations_packed, encoding = self.encoder(inp_packed, (h, c))  # encoding is just the last hidden state
        annotations = nn.utils.rnn.pad_packed_sequence(annotations_packed)[0]
        
        return (annotations, encoding), mask_src
        
    def get_loss(self):
        raise NotImplementedError()
        
    def forward(self, batch, **kwargs): # HACK
        if not self.training:
            return self.search(batch, **kwargs) 

        # encode
        annotations, mask_src = self.encode(batch)
        
        # decode
        loss = self.get_loss(batch, annotations, mask_src)
        return loss
        
class Seq2Seq(BaseModel):

    def __init__(self, **kwargs):
        super(Seq2Seq, self).__init__()
            
        self.cell_dim = kwargs['cell_dim']
        self.num_layers = kwargs['num_layers']
        self.src_vocab_size = kwargs['src_vocab_size']
        self.tgt_vocab_size = kwargs['tgt_vocab_size']
        self.MoE = kwargs['MoE']
        self.diversify = kwargs['diversify']
        self.num_experts = kwargs['num_experts']
        self.sampled_softmax = kwargs['sampled_softmax']
        self.num_samples = kwargs['num_samples']
        self.src_emb = nn.Embedding(self.src_vocab_size, self.cell_dim)
        self.tgt_emb = nn.Embedding(self.tgt_vocab_size, self.cell_dim)
        self.encoder = nn.LSTM(self.cell_dim, self.cell_dim, 
                               num_layers=self.num_layers,
                               bidirectional=True)
        self.attention = GlobalAttention(self.cell_dim, kwargs['dropout'])
        self.decoder = MultiLayerRNNCell(self.num_layers, 2 * self.cell_dim, self.cell_dim, module='LSTM', dropout=kwargs['dropout']) # bidirectional 

        if self.MoE:
            self.proj = MoEDecoder(self.cell_dim, self.num_experts, self.tgt_vocab_size, sampled_softmax=self.sampled_softmax, n_samples=self.num_samples)
        else:
            if self.sampled_softmax:
                self.proj = SampledSoftmax(self.tgt_vocab_size, self.num_samples, self.cell_dim)
            else:
                self.proj = nn.Linear(self.cell_dim, self.tgt_vocab_size)
        self.drop = nn.Dropout(kwargs['dropout'])

    def search(self, batch, **kwargs):
        annotations, mask_src = self.encode(batch)
        return self._forward(batch, annotations, mask_src)
        # return self.get_loss(batch, annotations, mask_src, test=True, record_alignment=True, **kwargs) # rename record_alignment
        # return None, loss

    def get_loss(self, batch, annotations, mask_src):
        return self._forward(batch, annotations, mask_src, teacher_forced=True, compute_loss=True)
        
    def _forward(self, batch, annotations, mask_src, 
                 teacher_forced=False,  
                 compute_loss=False): # flag to compute loss
        if self.training or compute_loss: assert teacher_forced
        
        annotations, encoding = annotations

        volatile = not self.training
        tl, bs = batch.input_dec.size()

        input_dec = self.tgt_emb(get_variable(batch.input_dec, volatile=volatile)) 
        target = get_variable(batch.target, volatile=volatile)
        mask_tgt = (target != PAD_ID).float()
        att = get_zeros([bs, self.cell_dim], training=self.training)
        state = self.decoder.init_state(bs)
        #state = self.decoder.init_state(bs, encoding=encoding)
        sl = batch.input_enc.size(0)
        if not teacher_forced:
            tl = int(1.5 * sl)
        att_helper = AttentionHelper(self.attention)
        src_tokens = get_variable(batch.input_enc)
        preds = list()
        alignments = list()
        losses = list()
        reg_loss = list()
        for j in xrange(tl):
            if j > 0 and not teacher_forced:
                cat = torch.cat([self.tgt_emb(preds[-1]), att], 1)
            else:
                cat = torch.cat([input_dec[j], att], 1)
            #cat = self.drop(cat) 
            state = self.decoder(cat, state)
            h_t = state.get_output()

            att, alignment = att_helper(annotations, h_t, mask_src)
            alignments.append(alignment.max(dim=1)[1])
            if self.MoE:
                if self.sampled_softmax:
                    labels = target[j] if self.training else None
                    expert_probs, all_logits = self.proj(self.drop(att), labels=labels)
                else:
                    expert_probs, all_logits = self.proj(self.drop(att))
                all_log_probs = nn.functional.log_softmax(all_logits, dim=2) # bs x ne x tvs
                if not self.training:
                    preds.append((all_log_probs.exp() * expert_probs.view(bs, self.num_experts, 1)).sum(dim=1).max(dim=1)[1])
                if compute_loss:
                    if self.sampled_softmax:
                        all_losses = all_log_probs[:, 0, :] # bs x ne
                    else:
                        all_losses = all_log_probs.gather(2, target[j].view(bs, 1, 1).expand(bs, self.num_experts, 1)).squeeze(dim=2) # bs x ne
                    losses.append((expert_probs * all_losses).sum(dim=1))
                    if self.diversify:
                        reg_loss.append(expert_probs)
            else:
                # get logits and log_probs
                if self.sampled_softmax:
                    labels = target[j] if self.training else None
                    logits = self.proj(self.drop(att), labels=labels)
                else:
                    logits = self.proj(self.drop(att))
                log_probs = nn.functional.log_softmax(logits) # bs x tvs
                
                # only retrieve predictions during testing/validation
                if not self.training:
                    preds.append(log_probs.max(dim=1)[1])
                
                # get loss
                if compute_loss:
                    if self.sampled_softmax:
                        losses.append(log_probs[:, 0])
                    else:
                        losses.append(log_probs.gather(1, target[j].view(-1, 1)).view(-1))
        
        if not self.training:
            preds = torch.stack(preds, 1) # bs x sl
        alignments = torch.stack(alignments, 1) # bs x sl
        if compute_loss:
            losses = -(torch.stack(losses, 0) * mask_tgt).sum() / len(batch)
            if self.diversify:
                expert_probs_masked = torch.stack(reg_loss, 0) * mask_tgt.unsqueeze(dim=2)
                expert_probs_masked_mean = expert_probs_masked.sum(dim=0).sum(dim=0) / mask_tgt.sum() # ne
                reg_loss = -(((expert_probs_masked - expert_probs_masked_mean) ** 2) * mask_tgt.unsqueeze(dim=2)).sum() / mask_tgt.sum() / self.num_experts
        
        res = VD([('alignments', alignments), ('losses', losses)])
        if self.diversify:
            res.append(('reg_loss', reg_loss))
        if not self.training:
            res.append(('predictions', preds))
        return res
        #return preds, alignments, losses
            
class Searcher(Seq2Seq):
    
    def __init__(self, **kwargs):
        super(Searcher, self).__init__(**kwargs)
        
        self.beam_width = kwargs['beam_width'] # beam width for testing 
        
    def init_beam_state(self, bs, bw):
        b_att = get_zeros([bs, bw, self.cell_dim])
        b_state = self.decoder.init_state(bs * bw).view(bs, bw, -1)
        b_score = get_variable(get_zeros([bs, bw], tensor=True).fill_(NEG_INF))
        b_prev_token = get_zeros([bs, bw]).long()
        b_prev_beam = get_zeros([bs, bw]).long()
        b_finished = get_variable(get_zeros([bs, bw], tensor=True).fill_(1).byte())
        b_alignment = get_zeros([bs, bw]).long()
        bs = Beam(False)
        bs.add_entry('attention', b_att)
        bs.add_entry('state', b_state)
        bs.add_entry('score', b_score)
        bs.add_entry('token', b_prev_token)
        bs.add_entry('from_beam', b_prev_beam)
        bs.add_entry('finished', b_finished)
        bs.add_entry('alignment', b_alignment)
        return bs

    def search(self, batch):
        # encode
        (annotations, encoding), mask_src = self.encode(batch)
        
        # search
        bs = len(batch)
        beam_state = self.init_beam_state(bs, self.beam_width)
        beam_state.finished[:, 0] = 0
        beam_state.token[:, 0] = SOS_ID
        beam_state.score[:, 0] = 0.0
        annotations = expand(annotations, 2, self.beam_width)
        mask_src = expand(mask_src, 2, self.beam_width)
        att_helper = AttentionHelper(self.attention)
        tl = int(1.5 * len(batch.input_enc))
        bs_ind = get_tensor(torch.arange(0, float(bs)).long()) 
        best = [(NEG_INF, None) for _ in xrange(bs)]
        record = Record()
        for j in xrange(tl):
            beam_state = self.next_step(beam_state, bs_ind, att_helper, mask_src, annotations)
            record.collect(beam_state)
        preds, alignments = record.find_best(batch, mask_src)
        return preds, alignments, None # NOTE None for losses
        
    def next_step(self, beam_state, bs_ind, att_helper, mask_src, annotations, 
                  K=None):
        if K is None:
            K = self.beam_width
            
        bs, bw = beam_state.token.size()
        prev_token = beam_state.token.view(bs * bw)
        att = beam_state.attention.view(bs * bw, -1)
        state = beam_state.state.view(bs * bw, -1)
        finished = beam_state.finished
        state = self.decoder(torch.cat([self.tgt_emb(prev_token), att], 1), state)
        h_t = state.get_output()
        att, alignment = att_helper(annotations, h_t, mask_src)
        most_aligned = alignment.max(dim=1)[1]
        scores = self.proj(att) # bsbw x tvs
        scores = scores.view(bs, bw, -1)
        scores = nn.functional.log_softmax(scores.view(bs * bw, -1)).view(bs, bw, -1) + NEG_INF * finished.float().unsqueeze(dim=2)
        scores = scores + beam_state.score.unsqueeze(dim=2)
        top_val, top_ind = scores.view(bs, -1).topk(K)
        token_id = top_ind % self.tgt_vocab_size
        beam_id = top_ind / self.tgt_vocab_size
        new_att = att.view(bs, bw, -1)[bs_ind.view(-1, 1), beam_id.data, :]
        new_state = state.view(bs, bw, -1)[bs_ind.view(-1, 1), beam_id.data, :]
        new_score = top_val
        new_prev_token = token_id
        new_prev_beam = beam_id
        new_finished = get_variable(finished.data[bs_ind.view(-1, 1), beam_id.data] | (token_id == EOS_ID).data)
        new_alignment = most_aligned.view(bs, bw)[bs_ind.view(-1, 1), beam_id.data]
        new_beam_state = Beam(False)
        new_beam_state.add_entry('attention', new_att, visible=False)
        new_beam_state.add_entry('state', new_state, visible=False)
        new_beam_state.add_entry('score', new_score)
        new_beam_state.add_entry('token', new_prev_token)
        new_beam_state.add_entry('from_beam', new_prev_beam)
        new_beam_state.add_entry('finished', new_finished)
        new_beam_state.add_entry('alignment', new_alignment)
        
        return new_beam_state
    
