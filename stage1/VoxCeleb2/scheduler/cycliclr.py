#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, lr_cyclic_min, lr_cyclic_max, lr_up_size, lr_down_size, lr_mode, **kwargs):

    lr_step = 'epoch'
    sche_fn = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_cyclic_min, max_lr=lr_cyclic_max, step_size_up=lr_up_size, step_size_down=lr_down_size, mode=lr_mode, cycle_momentum=False)
    print('Initialised cyclic LR scheduler')
    return sche_fn, lr_step
    #return sche_fn
