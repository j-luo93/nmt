from __future__ import division, print_function

import codecs
import subprocess
import sys
import time
import math
import cPickle
from collections import defaultdict

import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn as nn

from utils.helper import get_variable
from utils.vocab import EOS_ID, id2token

def create_optimizer(mod_or_params, lr, params=False):
    from optim import ScheduledOptim

    optimizer = ScheduledOptim(
        optim.Adam(
            mod_or_params.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        512, 4000)

    return optimizer


    params = mod_or_params.parameters() if not params else mod_or_params
    return optim.Adam(params, lr=lr)

# inplace translate
def translate_(translations, pred_ids, batch, vocab, src_tokens=None, dictionary=None, alignments=None):
    for idx, (i, ids) in enumerate(zip(batch.idx, pred_ids)): 
        assert i not in translations
        alignment = None if alignments is None else alignments[idx]
        translations[i] = id2token(ids, vocab, src_tokens=src_tokens[idx], dictionary=dictionary, alignments=alignment)

def bleu_test(translations, data_set, pred_path, gold_path=None):
    assert set(translations.keys()) == set(range(len(data_set)))
    with codecs.open(pred_path, 'w', 'utf8') as fbleu:
        for i in xrange(len(translations)):
            fbleu.write(translations[i] + '\n')
    try:
        out = subprocess.check_output('perl scripts/multi-bleu.perl -lc %s < %s' %(gold_path, pred_path), shell=True)
        score = float(out.split(',')[0].split('=')[1])
        print(out, end='', file=sys.stderr)
    except subprocess.CalledProcessError:
        print('No bleu score', file=sys.stderr)
        score = 0.0
    print('########################################') 
    return score

#################################################################################

class Tracker(object):
    
    def __init__(self, lr, cf, vf, name='global'):
        self.lr = lr
        self.cf = cf
        self.vf = vf
        self.step = 0
        self.epoch = 0
        self.epoch_started = False
        self.best_dev_loss = np.inf
        self.last_dev_loss = np.inf

        self.total_loss = 0.0
        self.total_norm = 0.0
        self.total_time = 0.0
        self.total_words = 0.0
        self.name = name
        
    def save(self, file_path):
        with open(file_path, 'wb') as fout:
            cPickle.dump(self.epoch, fout)
            cPickle.dump(self.epoch_started, fout)
            cPickle.dump(self.lr, fout)
            cPickle.dump(self.step, fout)
            cPickle.dump(self.best_dev_loss, fout)
            cPickle.dump(self.last_dev_loss, fout)
            
    def load(self, file_path):
        with open(file_path, 'rb') as fin:
            self.epoch = cPickle.load(fin)
            self.epoch_started = cPickle.load(fin)
            self.lr = cPickle.load(fin)
            self.step = cPickle.load(fin)
            self.best_dev_loss = cPickle.load(fin)
            self.last_dev_loss = cPickle.load(fin)
    
    def check_epoch_started(self):
        if not self.epoch_started:
            self.epoch_started = True
            self.epoch += 1

        print('################################################################################')
        print('\nepoch %d' %self.epoch, file=sys.stderr)
            
    
    def finish_epoch(self):
        self.epoch_started = False
        print('\nepoch %d finished' %self.epoch, file=sys.stderr)
        print('################################################################################')
        
    def start_timer(self):
        self.timer = time.time()
        
    def update(self, batch, updates, quiet=False):
        self.total_loss += updates['loss'] * len(batch)
        self.total_norm += updates['norm']
        
        self.total_words += batch.total_words
        self.step += 1

        self.total_time += time.time() - self.timer
        if not quiet: 
            print('\r%s step %d' %(self.name, self.step), end=' ')
            sys.stdout.flush()
    
    def check(self):
        loss = self.total_loss / self.total_words
        norm = self.total_norm / self.cf

        total_time = self.total_time / self.cf
        ppx = math.exp(loss) if loss < 300 else np.inf
        print('\r%s step %d, time per step %.4fs, loss %.4f, grad norm %.4f'
              %(self.name, self.step, total_time, loss, norm), file=sys.stderr)
        
        self.total_loss = 0.0
        self.total_norm = 0.0
        self.total_time = 0.0
        self.total_words = 0.0     

class TrainerBase(object):
    
    def __init__(self, model, **kwargs):
        self.data_dir = kwargs['data_dir']
        self.train_dir = kwargs['train_dir']
        self.src = kwargs['src']
        self.tgt = kwargs['tgt']
        self.bleu_path = kwargs['bleu_path']
        self.model = model
        self.optimizer = create_optimizer(self.model, kwargs['lr_init'])
        self.batch_size = kwargs['batch_size']
        self.num_epochs = kwargs['num_epochs']

    def validate(self, tracker, dev_set, vocab, gold_path, pred_path, trim=False, dictionary=None):
        self.model.eval()
        total_loss = 0.0
        total_words = 0.0
        translations = dict()
        translations = dict()
        risk = 0.0
        accuracy = 0.0
        for batch in dev_set: 
            self.model.zero_grad()
            res = self.model(batch) 
            preds = res.predictions
            alignments = res.alignments
            # NOTE imp
            preds = preds.data.cpu().numpy() 
            if alignments is not None:
                alignments = alignments.data.cpu().numpy() 
            
            translate_(translations, preds, batch, vocab, src_tokens=batch.src_tokens, dictionary=dictionary, alignments=alignments)
        print('\n########################################')
        dev_loss = -bleu_test(translations, dev_set, pred_path, gold_path=gold_path)
        print('Validation:\nnew/last/best %.4f/%.4f/%.4f' %(dev_loss, tracker.last_dev_loss, tracker.best_dev_loss), file=sys.stderr)
        
        # save
        if dev_loss < tracker.best_dev_loss:
            tracker.best_dev_loss = dev_loss
            torch.save(self.model.state_dict(), '%s/params.best' %self.train_dir)
        tracker.last_dev_loss = dev_loss
        torch.save(self.model.state_dict(), '%s/params.latest' %self.train_dir)
        tracker.save('%s/datastate.pkl' %self.train_dir)
        if tracker.epoch % 20 == 0:
            torch.save(self.model.state_dict(), '%s/params.%d' %(self.train_dir, tracker.epoch))
        
class Trainer(TrainerBase):
    
    def __init__(self, model, **kwargs):
        super(Trainer, self).__init__(model, **kwargs)
        
        self.tracker = Tracker(kwargs['lr_init'], kwargs['check_frequency'], kwargs['validate_frequency'])

    def train(self, train_set, dev_set, tgt_vocab, gold_path=None, dictionary=None): 
        if gold_path is None:
            gold_path = dev_set.files['tgt']
        steps_per_epoch = len(train_set) // self.batch_size
        while self.tracker.epoch < self.num_epochs or self.tracker.epoch_started:
            self.tracker.check_epoch_started()
            for i in xrange(steps_per_epoch):
                self.optimizer.zero_grad()
                self.tracker.start_timer()
                self.model.train()
                batch = train_set.get_next_batch()
                res = self.model(batch) 
                loss = res.loss
                loss.backward()
                
                updates = {'loss': loss.data[0]}
                updates['norm'] = nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                
                self.optimizer.step()
                
                self.tracker.update(batch, updates)
                
                if self.tracker.step % self.tracker.cf == 0: 
                    self.tracker.check()
                if self.tracker.vf > 0 and self.tracker.step % self.tracker.vf == 0:
                    self.validate(self.tracker, dev_set, tgt_vocab, gold_path, self.bleu_path, dictionary=dictionary)
            if self.tracker.vf == 0:
                self.validate(self.tracker, dev_set, tgt_vocab, gold_path, self.bleu_path, dictionary=dictionary)
                
            self.tracker.finish_epoch()
            
