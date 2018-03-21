from __future__ import print_function, division

import numpy as np
import torch
import math
import random
import cPickle
import os
import sys
from collections import namedtuple, defaultdict, Counter
from copy import deepcopy
import codecs

from utils.table import Table
from utils.vocab import PAD_ID, SOS_ID, EOS_ID, UNK_ID
from utils.helper import get_variable

#######################################################################################

def read_data(data_path, vocab, max_size=0):
    data = list()
    with codecs.open(data_path, 'r', 'utf8') as fin:
        cnt = 0
        for line in fin:
            tokens = line.strip().split()
            tokens = [vocab.get(token, UNK_ID) for token in tokens]
            data.append(tokens)
            if len(data) % 1000 == 0:
                print('\rread %d sentences' %len(data), end='')
                sys.stdout.flush()
            cnt += 1
            if max_size and cnt == max_size:
                break
    print('\rread %d sentences in total from %s' %(len(data), data_path))
    return data
                
def pad_sentence(sentence, length, EOS=False, SOS=False, reverse=False):
    padded = [SOS_ID] * SOS + sentence + [EOS_ID] * EOS
    if reverse:
        padded = list(reversed(padded))
    l = len(padded)
    assert l <= length, '%d: %d' %(l, length)
    return padded + [PAD_ID] * (length - l)

#######################################################################################

class Batch(Table):
    
    def __init__(self):
        super(Batch, self).__init__()

    def __len__(self):
        return len(self.real_src_length)
        
    def add_entry(self, name, value, dtype):
        assert dtype in ['int64', 'float32']
        value = np.asarray(value, dtype=dtype)
        super(Batch, self).add_entry(name, value, visible=True)

    def prepare(self):
        self._reindex()
        self._package()
    
    def _reindex(self):
        exemptions = set(['real_src_length'])
        for entry_name, entry in self.entry_dict.iteritems():
            if entry_name not in exemptions:
                self.entry_dict[entry_name] = np.transpose(entry)
    
    # wrap it in tensor, and possibly move to gpu
    def _package(self):
        use_cuda = (os.environ.get('USE_CUDA', None) == '1')
        for entry_name, entry in self.entry_dict.iteritems():
            packaged_entry = torch.from_numpy(entry).contiguous()
            if use_cuda:
                packaged_entry = packaged_entry.cuda()
            self.entry_dict[entry_name] = packaged_entry

################################################################################

# Take vocab and bucketed_corpus, and go!
class DataStreamBase(object):
    
    # corpus is bucketed
    def __init__(self, corpus, vocabs=None, **kwargs):
        self.corpus = corpus
        self.files = self.corpus.files
        
        self.batch_size = kwargs['batch_size']
        self.batch_first = kwargs['transformer']

    def __len__(self):
        return sum(map(len, self.corpus.indice_bins))
        
    def _get_next_batch_by_idx(self, idx):
        next_batch = self.corpus[idx] # a dict
        real_src_length = np.asarray(next_batch['src_length'])
        sort_indices = np.argsort(real_src_length)[::-1]
        real_src_length = real_src_length[sort_indices] + 1 # include EOS or SOS
        idx = idx[sort_indices]
        
        if 'tgt' in next_batch:
            input_enc, input_dec, target, weight = list(), list(), list(), list()
            if 'almt' in next_batch: 
                alignment = list()
                alignment_weight = list()
            size_enc_padded = max(next_batch['src_length']) + 1
            size_dec_padded = max(next_batch['tgt_length']) + 1
            
            for i in sort_indices:
                inp_e = next_batch['src'][i]
                inp_d = next_batch['tgt'][i]
                input_enc.append(pad_sentence(inp_e, size_enc_padded, EOS=True, reverse=False)) # No need to reverse it since we use bidirectional LSTM
                input_dec.append(pad_sentence(inp_d, size_dec_padded, SOS=True))
                target.append(pad_sentence(inp_d, size_dec_padded, EOS=True))
                weight.append(map(lambda x: 0.0 if x == PAD_ID else 1.0, target[-1]))
                if 'almt' in next_batch:
                    almt = next_batch['almt'][i] + [len(inp_e)] # align EOS with EOS
                    alignment.append(pad_sentence(almt, size_dec_padded))
                    #alignment_weight.append(map(lambda x: 1.0 if x != PAD else 0.0, alignment[-1]))
        else:
            input_enc = list()    
            size_enc_padded = max(next_batch['src_length']) + 1
            for i in sort_indices:
                inp_e = next_batch['src'][i]
                input_enc.append(pad_sentence(inp_e, size_enc_padded, EOS=True))
                
        # fill in data
        batch = Batch(batch_first=self.batch_first)
        batch.add_entry('idx', idx, 'int64')
        batch.add_entry('input_enc', input_enc, 'int64')
        batch.add_entry('real_src_length', real_src_length, 'int64')
        if 'tgt' in next_batch:
            batch.add_entry('input_dec', input_dec, 'int64')
            batch.add_entry('target', target, 'int64')
            batch.add_entry('weight', weight, 'float32')
            batch.add_metadata('total_words', int(batch.weight.sum()))
        if 'almt' in next_batch:
            batch.add_entry('alignment', alignment, 'int64')
            #batch.add_entry('alignment_weight', alignment_weight, 'float32')
        batch.prepare()
        # HACK
        batch.src_tokens = [next_batch['src_tokens'][si] for si in sort_indices]
        return batch

    def _get_next_batch_idx(self, idx):
        raise NotImplementedError
    
    def get_next_batch(self):
        idx = self._get_next_batch_idx()
        return self._get_next_batch_by_idx(idx)
        
class DataStream(DataStreamBase):
    
    def __iter__(self):
        n_bins = self.corpus.buckets.size
        if self.corpus.use_all:
            n_bins += 1
        self.bucket_ptr = 0
        self.bucket_batch_ptr = 0
        while self.bucket_ptr < n_bins:
            yield self.get_next_batch()
            
            self.bucket_batch_ptr += 1
            # use while not if to skip some empty buckets
            while self.bucket_batch_ptr * self.batch_size >= self.corpus.bin_sizes[self.bucket_ptr]:
                # attempt to update self.bucket_ptr now
                self.bucket_ptr += 1
                if self.bucket_ptr == n_bins:
                    break
                else:
                    self.bucket_batch_ptr = 0
    
    def _get_next_batch_idx(self):
        s = self.bucket_batch_ptr * self.batch_size
        t = min(s + self.batch_size, self.corpus.bin_sizes[self.bucket_ptr])
        ib = self.corpus.indice_bins[self.bucket_ptr]
        next_batch_idx = np.asarray([ib[i] for i in xrange(s, t)])
        return next_batch_idx
    
class DataStreamRandom(DataStreamBase):
    
    def __init__(self, corpus, **kwargs):
        super(DataStreamRandom, self).__init__(corpus, **kwargs)
        sizes = np.asarray(self.corpus.bin_sizes)
        self.pr = sizes / sizes.sum() 
    
    def _get_next_batch_idx(self):
        bucket_id = np.random.choice(len(self.pr), p=self.pr)
        next_batch_idx = np.random.choice(self.corpus.indice_bins[bucket_id], self.batch_size)
        return next_batch_idx

