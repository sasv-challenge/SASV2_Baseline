#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from adamp import AdamP

def Optimizer(parameters, lr, weight_decay, **kwargs):
    print('Initialised AdamP optimizer')
    return AdamP(parameters, lr = lr, betas = (0.9, 0.999), weight_decay = weight_decay)
