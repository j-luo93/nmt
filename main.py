from __future__ import print_function, division

from datetime import datetime
import pprint
import os
import cPickle
import sys

import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init

import argparse
import subprocess
import utils
from models import Seq2Seq, Searcher
from Transformer import Transformer
from trainer import bleu_test, translate_
from datastream import DataStream, DataStreamRandom
from trainer import Trainer

def prepare_corpus(names, paths, buckets, max_size=0, use_all=False, vocabs=None, method='both', replace_unk=False):
    corpus = utils.corpus.Corpus(max_size=max_size)
    # add columns
    for name, path in zip(names, paths):
        corpus.add_entry(name, path)
    # construct a dictionary
    dictionary = utils.vocab.Dictionary(corpus) if method == 'both' and replace_unk else None
    # indexify sentences
    if vocabs is not None:
        for k in vocabs:
            corpus.indexify(k, vocabs[k])
    # count lengths
    for entry_name in corpus.entry_dict.keys():
        if entry_name in ['src', 'tgt']:
            corpus.add_length_column(entry_name)
    # bucketify
    corpus = utils.corpus.BucketedCorpus(corpus, buckets, use_all=use_all, method=method)
    corpus.dictionary = dictionary # HACK
    return corpus

def train(args):
    train_src_data_path = '%s/train.%s' %(args.data_dir, args.src)
    train_tgt_data_path = '%s/train.%s' %(args.data_dir, args.tgt)
    names = ['src', 'tgt']
    paths = [train_src_data_path, train_tgt_data_path]
    if args.replace_unk:
        names.append('almt')
        paths.append('%s/train.align' %(args.data_dir))
    train_corpus = prepare_corpus(names,
                                  paths,
                                  buckets, 
                                  max_size=args.max_size, 
                                  vocabs=vocabs,
                                  replace_unk=args.replace_unk)
    
    dictionary = train_corpus.dictionary 
    dev_src_data_path = '%s/dev.%s' %(args.data_dir, args.src) 
    dev_tgt_data_path = '%s/dev.%s' %(args.data_dir, args.tgt)
    if args.DEBUG: # use smaller dev set for debugging 
        if not os.path.isfile(dev_src_data_path + '.debug'):
            subprocess.call('head -n 100 %s > %s' %(dev_src_data_path, dev_src_data_path + '.debug'), shell=True)
            subprocess.call('head -n 100 %s > %s' %(dev_tgt_data_path, dev_tgt_data_path + '.debug'), shell=True)
        dev_src_data_path += '.debug'
        dev_tgt_data_path += '.debug'
        
    dev_corpus = prepare_corpus(['src', 'tgt'],
                                [dev_src_data_path, dev_tgt_data_path], 
                                buckets, 
                                max_size=args.max_size, 
                                use_all=True, # use all data for dev, which means do NOT discard very long sentences
                                vocabs=vocabs,
                                method='src')
    
    train_set = DataStreamRandom(train_corpus, vocabs=vocabs, **vars(args)) # random batching
    dev_set = DataStream(dev_corpus, vocabs=vocabs, **vars(args)) # no random batching, iterated by length
    trainer = Trainer(model, **vars(args))
    
    datastate_path = '%s/datastate.pkl' %args.train_dir 
    if os.path.isfile(datastate_path):
       trainer.tracker.load(datastate_path) 
       
    trainer.train(train_set, dev_set, tgt_vocab, dictionary=dictionary)
    
def test(args):
    train_src_data_path = '%s/train.%s' %(args.data_dir, args.src)
    train_tgt_data_path = '%s/train.%s' %(args.data_dir, args.tgt)
    names = ['src', 'tgt']
    paths = [train_src_data_path, train_tgt_data_path]
    if args.replace_unk:
        names.append('almt')
        paths.append('%s/train.align' %(args.data_dir))
    train_corpus = prepare_corpus(names,
                                  paths,
                                  buckets, 
                                  max_size=args.max_size, 
                                  vocabs=vocabs,
                                  replace_unk=args.replace_unk)
    
    dictionary = train_corpus.dictionary 

    if args.test_path is not None:
        test_src_data_path = '%s.%s' %(args.test_path, args.src) 
        #test_tgt_data_path = '%s.%s' %(args.test_path, args.tgt)
    else:
        test_src_data_path = '%s/test.%s' %(args.data_dir, args.src) 
        #test_tgt_data_path = '%s/test.%s' %(args.data_dir, args.tgt)

    test_corpus = prepare_corpus(['src'], [test_src_data_path], buckets, vocabs={'src': vocabs['src']}, use_all=True, method='src') 
    test_set = DataStream(test_corpus, vocabs=vocabs, **vars(args))
    
    model.eval()
    translations = dict()
    alignments = dict()
    start = time.time()
    for b, batch in enumerate(test_set, 1):
        print('\rbatch %d: %.4fs' %(b, time.time() - start), end='')
        sys.stdout.flush()
        
        preds, alignments, _ = model.search(batch) 
        translate_(translations, preds, batch, vocabs['tgt'], src_tokens=batch.src_tokens, dictionary=dictionary, alignments=alignments)
    print()
    bleu_test(translations, test_set, args.bleu_path, gold_path=None) # NOTE no bleu evaluation for test
        
def init_params(model):
    for p in model.state_dict().itervalues():
        nn.init.uniform(p, a=-0.08, b=0.08)
    for name, p in model.named_parameters():
        if 'bias_ih' in name or 'bias_hh' in name:
            size = p.size(0)
            ind = torch.arange(size // 4, size // 2).long()
            p.data[ind] = 1.0
            
def load_params(load_path, model):
    saved_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
    state_dict = model.state_dict()
    # NOTE only update params existent in model
    for k in saved_dict:
        if k in state_dict:
            state_dict[k] = saved_dict[k]
    model.load_state_dict(state_dict)
    print('Loaded parameters from %s' %load_path)
    
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', type=str, help='mode')
    parser.add_argument('data_dir', type=str, help='Data directory')
    parser.add_argument('-DEBUG', action='store_true', help='Debug mode')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed')
    parser.add_argument('-random', '-r', action='store_true', help='use random random seed')
    parser.add_argument('-cuda', '-c', action='store_true', help='Use cuda')
    parser.add_argument('--beam_width', '-bw', default=1, type=int, help='Beam width', metavar='')
    parser.add_argument('-replace_unk', '-ru', action='store_true', help='Replace unknown tokens')
    # arguments for training
    train_group = parser.add_argument_group('Train')
    train_group.add_argument('--num_epochs', '-ne', default=50, type=int, help='Number of epochs', metavar='')
    train_group.add_argument('--batch_size', '-bs', default=64, type=int, help='Batch size', metavar='')
    train_group.add_argument('--max_length', '-ml', default=100, type=int, help='Maximum sequence length', metavar='')
    train_group.add_argument('--max_size', '-ms', default=0, type=int, help='Maximum training data size', metavar='')
    train_group.add_argument('--check_frequency', '-cf', default=100, type=int, help='How ofter to check progress', metavar='')
    train_group.add_argument('--validate_frequency', '-vf', default=0, type=int, help='How ofter to validate', metavar='')
    train_group.add_argument('--msg', '-M', type=str, help='Message', metavar='')
    train_group.add_argument('--lr_init', default=0.0005, type=float, help='Initial learning rate', metavar='')
    train_group.add_argument('--dropout', '-do', default=0.5, type=float, help='dropout rate', metavar='')
    train_group.add_argument('--src', '-s', default='de', type=str, help='Source language', metavar='')
    train_group.add_argument('--tgt', '-t', default='en', type=str, help='Target language', metavar='')
    train_group.add_argument('--model_path', '-mp', type=str, help='Path to trained model', metavar='')
    # arguments for testing
    test_group = parser.add_argument_group('Test')
    test_group.add_argument('--test_path', '-tp', help='test path', metavar='')
    # arguments for model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--cell_dim', '-cd', default=256, type=int, help='Cell dimensionality', metavar='')
    model_group.add_argument('--num_layers', '-nl', default=1, type=int, help='Number of layers', metavar='')
    model_group.add_argument('--src_vocab_size', '-svs', default=30000, type=int, help='Vocabulary size for source language', metavar='')
    model_group.add_argument('--tgt_vocab_size', '-tvs', default=15000, type=int, help='Vocabulary size for target language', metavar='')
    model_group.add_argument('-Transformer', '-T', action='store_true', help='Use Transformer architecture')
    model_group.add_argument('--num_heads', '-nh', default=8, type=int, help='Number of heads for mult-head attention', metavar='')

    args = parser.parse_args()
    
    if args.cuda:
        os.environ['USE_CUDA'] = '1'

    if args.DEBUG:
        args.batch_size = 2
        args.max_size = 10000
        args.cell_dim = 6
        args.src_vocab_size = 500
        args.tgt_vocab_size = 500
        
    assert args.mode in ['train', 'test'], args.mode
    
    # set up training directory and temporary files
    now = datetime.now()
    date = now.strftime("%m-%d")
    timestamp = now.strftime("%H:%M:%S")
    if not args.msg:
        args.train_dir = 'train/%s/%s' %(date, timestamp)
    else:
        args.train_dir = 'train/%s/%s-%s' %(date, args.msg, timestamp)
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)
    args.bleu_path = '%s/bleu' %args.train_dir

    return args
    
if __name__ == '__main__':
    args = parse_args()
    
    buckets = utils.corpus.Buckets(args.max_length, 5)
    sys.stderr = utils.logger.Logger(args.train_dir + '/log')
    
    src_vocab_path = '%s/vocab.%s' %(args.data_dir, args.src)
    tgt_vocab_path = '%s/vocab.%s' %(args.data_dir, args.tgt)
    src_vocab = utils.vocab.Vocab(src_vocab_path, 0, threshold=2)
    tgt_vocab = utils.vocab.Vocab(tgt_vocab_path, args.tgt_vocab_size, threshold=2, replace_unk=args.replace_unk)
    vocabs = {'src': src_vocab, 'tgt': tgt_vocab}
    
    # change vocab size after thresholdding
    args.src_vocab_size = len(src_vocab)
    args.tgt_vocab_size = len(tgt_vocab)
    
    if args.mode == 'train':
        if args.Transformer:
            model_mod = Transformer
        else:
            model_mod = Seq2Seq
    elif args.mode == 'test':
        model_mod = Searcher 
    else:
        raise
    model = model_mod(**vars(args))
    init_params(model)
        
    # load trained model    
    if args.mode == 'test':
        load_params(args.model_path + '/params.best', model)

    if args.cuda:
        print('Getting things set up on gpu...', end='')
        sys.stdout.flush()
        model.cuda()
        print('Done')
    print(pprint.pformat(model), file=sys.stderr)    
    print(pprint.pformat(vars(args)), file=sys.stderr)

    if not args.random:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    torch.set_printoptions(precision=4, threshold=50)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
