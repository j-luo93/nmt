from __future__ import division, print_function

import sys

from table import Table
from vocab import UNK_ID

import codecs

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
    def add_entry(self, name, value_source):
        assert name in ['src', 'tgt', 'almt'], '%s not supported' %name
        
        self.files[name] = value_source
        with codecs.open(value_source, 'r', 'utf8') as fin:
            value = list()
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
                    value.append(line.strip().split())
                if len(value) % 1000 == 0:
                    print('\rRead %d sentences from %s' %(len(value), value_source), end='')
                    sys.stdout.flush()
                if len(value) == self.max_size:
                    break
        print('\rRead %d sentences from %s in total' %(len(value), value_source))
        super(Corpus, self).add_entry(name, value, visible=False)
    
    def indexify(self, name, vocab):
        new_entry = list()
        for e_i, entry in enumerate(self.entry_dict[name]):
            idx = list()
            for i, w in enumerate(entry):
                iw = vocab[w]
                idx.append(iw)
            new_entry.append(idx)
        if name == 'src':
            self.entry_dict['src_tokens'] = self.entry_dict['src']
            self.entry_dict['src'] = new_entry
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
