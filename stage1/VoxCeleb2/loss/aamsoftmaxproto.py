#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import loss.aamsoftmax as aamsoftmax
import loss.angleproto as angleproto

class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()
        self.test_normalize = True
        self.aamsoftmax = aamsoftmax.LossFunction(**kwargs)
        self.angleproto = angleproto.LossFunction(**kwargs)
        print('Initialised AAMSoftmaxPrototypicalLoss')

    def forward(self, x, label=None):
        assert x.size()[1] == 2
        nlossS, prec1 = self.aamsoftmax(x.reshape(-1,x.size()[-1]), label.repeat_interleave(2))
        nlossM, _     = self.angleproto(x,None)
        return nlossS+nlossM, prec1
