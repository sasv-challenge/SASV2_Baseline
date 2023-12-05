#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, **kwargs):

    sche_fn = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    lr_step = 'epoch'
    print('Initialised exponential LR scheduler')
    return sche_fn, lr_step
    #return sche_fn
