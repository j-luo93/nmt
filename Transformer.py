from __future__ import division, print_function

import cPickle
import numpy as np
import torch
import torch.nn as nn

from utils.helper import get_variable, get_zeros, expand, get_tensor, where, get_values
from utils.vocab import PAD_ID, SOS_ID, EOS_ID
from utils.variable_dict import VariableDict as VD
from modules import PositionalEncoding, MaskedNLLLoss, EncoderStack, DecoderStack
from models_utils import Record, NEG_INF, cat, AttentionHelper, Beam

class Transformer(nn.Module):
        
    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        
        self.cell_dim = kwargs['cell_dim']
        self.num_heads = kwargs['num_heads'] # TODO naming: n_heads or num_heads
        self.num_layers = kwargs['num_layers']
        self.src_vocab_size = kwargs['src_vocab_size']
        self.tgt_vocab_size = kwargs['tgt_vocab_size']
        
        # declare modules
        self.src_emb = nn.Embedding(self.src_vocab_size, self.cell_dim)
        self.tgt_emb = nn.Embedding(self.tgt_vocab_size, self.cell_dim)
        self.position_encoder = PositionalEncoding(self.cell_dim)
        self.encoder = EncoderStack(self.cell_dim, self.num_heads, self.num_layers)
        self.decoder = DecoderStack(self.cell_dim, self.num_heads, self.num_layers)
        self.projection = nn.Linear(self.cell_dim, self.tgt_vocab_size)
        self.output = MaskedNLLLoss()
        
    def forward(self, batch):
        bs, sl = batch.input_enc.size()
        _, tl = batch.input_dec.size()
        if self.training:
            input_enc_sym = get_variable(batch.input_enc)
            input_dec_sym = get_variable(batch.input_dec)
            mask_tgt = get_variable(batch.weight)
            input_enc_emb = self.src_emb(input_enc_sym)
            input_dec_emb = self.src_emb(input_dec_sym)
            pos_enc = self.position_encoder(get_variable(torch.arange(0, sl).long().expand_as(input_enc_sym)))
            pos_dec = self.position_encoder(get_variable(torch.arange(0, tl).long().expand_as(input_dec_sym)))
            input_enc = input_enc_emb + pos_enc
            input_dec = input_dec_emb + pos_dec
            encoder_states = self.encoder(input_enc)
            decoder_states = self.decoder(input_dec, encoder_states)
            
            logits = self.projection(decoder_states) # take the output from the last decoder layer 
            targets = get_variable(batch.target)
            loss = self.output(logits, targets, mask=mask_tgt).sum() / len(batch)
            res = VD([('loss', loss)])
            return res
        else:
            input_enc_sym = get_variable(batch.input_enc, volatile=True)
            input_enc_emb = self.src_emb(input_enc_sym)
            pos_enc = self.position_encoder(get_variable(torch.arange(0, sl).long().expand_as(input_enc_sym), volatile=True))
            input_enc = input_enc_emb + pos_enc
            encoder_states = self.encoder(input_enc_emb)
            tl = int(1.5 * input_enc.size(1))
            pos_dec = self.position_encoder(get_variable(torch.arange(0, tl).long().view(1, tl).expand(bs, tl), volatile=True))
            input_dec_emb = self.tgt_emb(get_values([bs], SOS_ID, volatile=True).long())
            predictions = list()
            dec_emb_list = list()
            for i in xrange(tl):
                input_dec = (input_dec_emb + pos_dec[:, i]).unsqueeze(dim=1)
                dec_emb_list.append(input_dec)
                decoder_states = self.decoder(torch.cat(dec_emb_list, dim=1), encoder_states)[:, -1, :]
                logits = self.projection(decoder_states)
                pred = logits.max(dim=1)[1]
                predictions.append(pred)
                input_dec_emb = self.tgt_emb(pred)
            predictions = torch.stack(predictions, dim=1)
            res = VD([('predictions', predictions)])
            return res
        
