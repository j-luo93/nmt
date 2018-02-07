import torch
import math
import torch.nn as nn
from copy import deepcopy

from utils.helper import get_zeros, get_variable, where, get_tensor, get_values, tanh, NEG_INF
from utils.vocab import EOS_ID
from utils.LSTMState import LSTMState
from utils.table import Table

def cat(gold, beam, dim):
    size = list(gold.size())
    size.insert(dim, 1)
    if isinstance(gold, LSTMState):
        return gold.view(*size).cat(beam, dim)
    else:
        return torch.cat([gold.view(*size), beam], dim)

class AttentionHelper(object):
    
    def __init__(self, att_mod):
        self.precomputed = False
        self.att_mod = att_mod
        self.hierarchical = False # HACK
    
    def __call__(self, h_s, h_t, mask_src, **kwargs):
        if not self.precomputed:
            if hasattr(self.att_mod, 'precompute'):
                self.Wh_s = self.att_mod.precompute(h_s, **kwargs)
            self.precomputed = True
                    
        return self.att_mod(self.Wh_s, h_t, mask_src, h_s, **kwargs)
            
    def combine(self, context, h_t):
        cat = torch.cat([context, h_t], 1)
        cat = self.att_mod.drop(cat) # NOTE dropout
        h_tilde = tanh(self.att_mod.hidden(cat))
        return h_tilde

class Record(object):
    
    def __init__(self):
        self.scores = list()
        self.tokens = list()
        self.from_beams = list()
        self.finished = list()
        self.alignments = list()
    
    def collect(self, beam):
        self.scores.append(beam.score)
        self.tokens.append(beam.token)
        self.from_beams.append(beam.from_beam)
        self.finished.append(beam.finished)
        self.alignments.append(beam.alignment)
    
    def find_best(self, batch, mask_src):
        bs, bw = self.scores[-1].size()
        tl = len(self.scores)
        scores = torch.stack(self.scores, 0).data.cpu().numpy()
        tokens = torch.stack(self.tokens, 0).data.cpu().numpy()
        beams = torch.stack(self.from_beams, 0).data.cpu().numpy()
        finished = torch.stack(self.finished, 0).data.cpu().numpy()
        all_alignments = torch.stack(self.alignments, 0).data.cpu().numpy()
        
        preds = list()
        alignments = list()
        orig_sents = batch.input_enc.cpu()
        for batch_i in xrange(bs):
            best = (NEG_INF, None, None)
            subseq = [list() for _ in xrange(bw)]
            subalign = [list() for _ in xrange(bw)]
            orig_tokens = orig_sents[:, batch_i] 
            sl = batch.input_enc.size(0)
            for r_i in xrange(tl):
                new_subseq = [list() for _ in xrange(bw)]
                new_subalign = [list() for _ in xrange(bw)]
                for beam_i in xrange(bw):
                    new_subseq[beam_i] = subseq[beams[r_i, batch_i, beam_i]] + [tokens[r_i, batch_i, beam_i]]
                    new_subalign[beam_i] = subalign[beams[r_i, batch_i, beam_i]] + [all_alignments[r_i, batch_i, beam_i]]
                    score = scores[r_i, batch_i, beam_i]

                    check, remove = self.check(new_subseq[beam_i], r_i == tl - 1)
                    if check:
                        x = new_subseq[beam_i]
                        y = new_subalign[beam_i]
                        if remove:
                            x = x[:-1]
                            y = y[:-1]
                        full_score = score / (r_i + 1)
                        if full_score > best[0]:
                            best = (full_score, x, y)
                subseq = new_subseq
                subalign = new_subalign
            preds.append(best[1])
            alignments.append(best[2])
        return preds, alignments

    def check(self, seq, end):
        remove = seq[-1] == EOS_ID
        if end:
            return all(map(lambda x : x != EOS_ID, seq[:-1])), remove
        if remove:
            return all(map(lambda x : x != EOS_ID, seq[:-1])), remove
        return False, False

class Beam(Table):
    
    def __init__(self, gold, mid=False):
        super(Beam, self).__init__()
        self.gold = gold
        self.mid = mid
    
    def cat(self, other):
        assert self.gold and not other.gold
        new_beam = Beam(False)
        for entry_name in self.entry_dict:
            new_entry = cat(self[entry_name], other[entry_name], 1)
            visible = entry_name in self.visible_entry_list
            new_beam.add_entry(entry_name, new_entry, visible=visible)
        return new_beam
    
    def consolidate(self, other, choose_other):
        assert not self.gold and not other.gold
        new_beam = Beam(False)
        for entry_name in self.entry_dict:
            dim = self.entry_dict[entry_name].dim()
            if dim == 2:
                mask = choose_other.view(-1, 1)
            else:
                mask = choose_other.view(-1, 1, 1)
            new_entry = where(mask, other[entry_name], self[entry_name])
            visible = entry_name in self.visible_entry_list
            new_beam.add_entry(entry_name, new_entry, visible=visible)
        return new_beam
    
    def sub(self):
        assert not self.gold
        size = self.token.size(1)
        assert size > 1
        
        new_beam = Beam(False)
        for entry_name in self.entry_dict:
            dim = self.entry_dict[entry_name].dim()
            assert dim > 1
            entry = self.entry_dict[entry_name][:, :-1].contiguous()
            visible = entry_name in self.visible_entry_list
            new_beam.add_entry(entry_name, entry, visible=visible)
        return new_beam
