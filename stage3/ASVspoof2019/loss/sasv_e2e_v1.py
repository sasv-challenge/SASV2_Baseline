#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.aamsoftmax as aamsoftmax
import loss.angleproto_sasv as angleproto_sasv

class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.aamsoftmax = aamsoftmax.LossFunction(**kwargs)
        self.angleproto_sasv = angleproto_sasv.LossFunction(**kwargs)
        self.num_class = kwargs.get('num_class')
        print('Initialized SASV End-to-end v1 Loss Function')

    def forward(self, x, label=None):
        assert x.size()[1] == 2
        nlossS, prec = self.aamsoftmax(x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))
        
        idx_bna = torch.where(label != self.num_class-1)
        idx_spf = torch.where(label == self.num_class-1)
        x1 = x[idx_bna]
        x2 = x[idx_spf]
        x = torch.cat((x1, x2))
        nlossM, _ = self.angleproto_sasv(x, None, len(idx_bna[0]))
        
        return nlossS + nlossM, prec
