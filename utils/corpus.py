from __future__ import division, print_function

import sys

from table import Table
from vocab import UNK_ID, POS_UNK_IDs, _PAD, PAD_ID

import codecs

MAX_MORPH = 5 # maximum amount of morphemes per token

class Buckets(object):
    
    def __init__(self, max_length, step_size):
        self.max_length = max_length
        self.step_size = step_size
        assert self.max_length % self.step_size == 0
        self.size = self.max_length // self.step_size
    
    def get_bucket_id(self, length):
        if length > self.max_length:
            return self.size
        id_ = length // self.step_size - 1
        if length % self.step_size > 0:
            id_ += 1
        return id_

class Corpus(Table):
    
    def __init__(self, max_size=0):
        super(Corpus, self).__init__()
        self.max_size = max_size
        self.files = dict() # where we get the sentences
        
    # add entry from value source file, instead of directly from value
    def add_entry(self, name, value_source, morph=False):
        assert name in ['src', 'tgt', 'almt'], '%s not supported' %name
        
        self.files[name] = value_source
        
        value = list()
        morph_value = list()
        cnt_too_long = 0
        with codecs.open(value_source, 'r', 'utf8') as fin:
            for i, line in enumerate(fin):
                if name == 'almt':
                    parts = line.strip().split()
                    almt = [-1] * len(parts)
                    for part in parts:
                        s, t = map(int, part.split('-'))
                        try:
                            almt[t] = s
                        except:
                            import ipdb; ipdb.set_trace()
                    assert all(map(lambda x: x != -1, almt))
                    value.append(almt)
                else:
                    if morph:
                        tokens = line.strip().split()
                        segs = list()
                        for token in tokens:
                            orig_segs = token.split('-')
                            if len(orig_segs) > MAX_MORPH:
                                cnt_too_long += 1
                                orig_segs = orig_segs[:MAX_MORPH]
                            elif len(orig_segs) < MAX_MORPH:
                                orig_segs = orig_segs + [_PAD] * (MAX_MORPH - len(orig_segs))
                            segs.append(orig_segs)
                        value.append(tokens)
                        morph_value.append(segs)
                    else:
                        value.append(line.strip().split())
                if len(value) % 1000 == 0:
                    print('\rRead %d sentences from %s' %(len(value), value_source), end='')
                    sys.stdout.flush()
                if len(value) == self.max_size:
                    break
            print('\rRead %d sentences from %s in total' %(len(value), value_source))
        
        super(Corpus, self).add_entry(name, value, visible=False)
        if len(morph_value) > 0:
            super(Corpus, self).add_entry('morph', morph_value, visible=False)
            print('%d tokens with too long morphemes' %cnt_too_long)
            
    def indexify(self, name, vocab):
        new_entry = list()
        new_weight_entry = list()
        for e_i, entry in enumerate(self.entry_dict[name]):
            idx = list()
            morph_weights = list() # HACK compute morph weights here
            for i, w in enumerate(entry):
                if name == 'morph':
                    idx_entry = list()
                    weight_entry = list()
                    for t in w:
                        it = vocab[t]
                        idx_entry.append(it)
                        weight_entry.append(float(it != PAD_ID))
                    idx.append(idx_entry)
                    morph_weights.append(weight_entry)
                else:
                    iw = vocab[w]
                    idx.append(iw)
            new_entry.append(idx)
            new_weight_entry.append(morph_weights)
        if name == 'src':
            self.entry_dict['src_tokens'] = self.entry_dict['src']
            self.entry_dict['src'] = new_entry
        elif name == 'morph':
            self.entry_dict['morph'] = new_entry
            self.entry_dict['morph_weight'] = new_weight_entry
        else:
            self.entry_dict[name] = new_entry
    
    def add_length_column(self, name):
        assert name in self.entry_dict
        value = map(len, self.entry_dict[name])
        super(Corpus, self).add_entry(name + '_length', value, visible=False)
    
class BucketedCorpus(object):
    
    def __init__(self, corpus, buckets, use_all=False, method='both'):
        self.corpus = corpus
        self.buckets = buckets
        self.use_all = use_all
        n_bins = self.buckets.size
        if use_all: n_bins += 1
        self.indice_bins = [list() for _ in xrange(n_bins)]
        
        assert method in ['both', 'src']
        
        for i in xrange(len(corpus)):
            src = corpus['src'][i]
            id_src = self.buckets.get_bucket_id(len(src))
            if method == 'both':
                tgt = corpus['tgt'][i]
                id_tgt = self.buckets.get_bucket_id(len(tgt))
                id_ = max(id_src, id_tgt)
            else:
                id_ = id_src
            if id_ < n_bins:
                self.indice_bins[id_].append(i)
        self.bin_sizes = [len(b) for b in self.indice_bins]
        self.sources = sorted(self.corpus.entry_dict.keys())
        self.files = self.corpus.files
    
    # get row(s) 
    def __getitem__(self, key):
        return {s: [self.corpus[s][k] for k in key] for s in self.sources}
