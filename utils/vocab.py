from __future__ import division, print_function

import os
import argparse
import sys
import codecs
from collections import defaultdict, Counter
from itertools import izip
from operator import itemgetter

# Special vocabulary symbols - we always put them at the start.
_PAD = '_PAD'
_SOS = '_SOS'
_EOS = '_EOS'
_UNK = '_UNK'
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

_POSUNK = ['_UNK_%d' %p for p in xrange(-7, 8)]
_START_VOCAB_TGT = [_PAD, _SOS, _EOS, _UNK] + _POSUNK

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2 
UNK_ID = 3
POS_UNK_IDs = [11 + x for x in xrange(-7, 8)]

def prepare_vocab(data_path, vocab_path, size, threshold=0, morph=True):
    if os.path.isfile(vocab_path): 
        print('Vocabulary file already exists.', file=sys.stderr)
        return
    else:
        print('Creating vocabulary from %s...' %data_path, file=sys.stderr)
        with codecs.open(data_path, 'r', 'utf8') as fin, codecs.open(vocab_path, 'w', 'utf8') as fout:
            cnt = Counter()
            for line in fin:
                tokens = line.strip().split()
                for t in tokens:
                    if morph:
                        for tt in t.split('|'):
                            cnt[tt] += 1
                    else:
                        cnt[t] += 1
            cnt = dict(filter(lambda x: x[1] >= threshold, cnt.iteritems()))
            vocab = sorted(cnt.iteritems(), key=itemgetter(1), reverse=True)
            if size > 0:
                vocab = vocab[:size]
            print('Vocab size: %d' %len(vocab))
            fout.write('%d\n' %len(vocab))
            for k, v in vocab:
                fout.write('%s\t%d\n' %(k, v))

def id2token(tgt_ids, tgt_vocab, src_tokens=None, dictionary=None, alignments=None):
    if alignments is not None and tgt_vocab.replace_unk:
        assert src_tokens is not None and dictionary is not None
        return id2tokenWithAlignments(tgt_ids, tgt_vocab, alignments, src_tokens, dictionary)
    # elif tgt_vocab.pos_unk:
    #     assert src_tokens is not None and dictionary is not None
    #     return id2tokenWithDict(tgt_ids, tgt_vocab, src_tokens, dictionary)
    else:
        return id2tokenBasic(tgt_ids, tgt_vocab)
    
def id2tokenWithAlignments(tgt_ids, tgt_vocab, alignments, src_tokens, dictionary):
    translations = list()                              
    for i, id_ in enumerate(tgt_ids):
        if id_ == EOS_ID:
            break
        else:
            if id_ >= len(tgt_vocab):
                translations.append(_UNK)
            else:
                wt = tgt_vocab[id_]
                if '_UNK' in wt:
                    if alignments[i] == len(src_tokens): # aligned to EOS # HACK this shouldn't happen
                        wt = _EOS
                    else:
                        ws = src_tokens[alignments[i]]
                        wt = dictionary[ws]
                translations.append(wt)
    return ' '.join(translations)


def id2tokenWithDict(tgt_ids, tgt_vocab, src_tokens, dictionary): # TODO use python style naming
    translations = list()
    for i, id_ in enumerate(tgt_ids):
        if id_ == EOS_ID:
            break
        else:
            if id_ >= len(tgt_vocab):
                translations.append(_UNK)
            elif 4 <= id_ <= 18:
                rel_pos = id_ - 11
                j = i + rel_pos
                if 0 <= j < len(src_tokens):
                    ws = src_tokens[j]
                    wt = dictionary[ws]
                    translations.append(wt)
                else:
                    translations.append(_UNK)
            else:
                translations.append(tgt_vocab[id_])
    return ' '.join(translations)
    
def id2tokenBasic(tgt_ids, tgt_vocab):
    translations = list()
    for id_ in tgt_ids:
        if id_ == EOS_ID:
            break
        else:
            if id_ >= len(tgt_vocab):
                translations.append(_UNK)
            else:
                translations.append(tgt_vocab[id_])
    return ' '.join(translations)
        
################################################################################

class Vocab(object):
    
    def __init__(self, vocab_path, size, replace_unk=False, threshold=None):
        self.w2i = dict(zip(_START_VOCAB, range(len(_START_VOCAB))))
        self.i2w = list(_START_VOCAB)
        self.replace_unk = replace_unk
            
        with codecs.open(vocab_path, 'r', 'utf8') as fin:
            cached_size = int(fin.readline())
            if threshold is None:
                print('Asking for %d, from %d cached' %(size, cached_size), file=sys.stderr)
            else:
                print('Vocab frequency threshold %d, from %d cached' %(threshold, cached_size), file=sys.stderr)
            for line in fin:
                word, freq = line.strip().split('\t')
                freq = int(freq)
                if threshold is None or freq >= threshold:
                    self.w2i[word] = len(self.w2i)
                    self.i2w.append(word)
                    if len(self.w2i) == size:
                        break
        print('Vocab size: %d' %len(self), file=sys.stderr)
        
    def __len__(self):
        return len(self.w2i)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.i2w[key]
        elif isinstance(key, unicode) or isinstance(key, str):
            return self.w2i.get(key, UNK_ID) 
        else:
            raise

class Dictionary(object):
    
    def __init__(self, corpus):
        assert set(['src', 'tgt', 'almt']) <= set(corpus.files.keys())
        
        self.dict_ = defaultdict(Counter)
        for s, t, a in izip(corpus['src'], corpus['tgt'], corpus['almt']):
            for ti, si in enumerate(a):
                ws = s[si]
                wt = t[ti]
                self.dict_[ws][wt] += 1
        for w in self.dict_.iterkeys():
            best = max(self.dict_[w], key=self.dict_[w].get)
            self.dict_[w] = best
        self.dict_ = dict(self.dict_) # make it a plain dict
    
    def __getitem__(self, key):
        return self.dict_.get(key, key) # fall back to identity translation if not in dict

################################################################################

if __name__  == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', '-dp', metavar='')
    parser.add_argument('--vocab_path', '-vp', metavar='')
    parser.add_argument('--size', '-s', type=int, metavar='')
    parser.add_argument('--threshold', '-t', type=int, default=3, metavar='')
    args = parser.parse_args()
    
    prepare_vocab(args.data_path, args.vocab_path, args.size, threshold=args.threshold)
