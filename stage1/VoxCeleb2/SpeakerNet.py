#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import time
import itertools
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import SWALR, AveragedModel


class WrappedModel(nn.Module):
    
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    
    def __init__(self, model, optimizer, trainfunc, num_utt, **kwargs):
        super(SpeakerNet, self).__init__()
        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs)
        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs)
        self.num_utt = num_utt

    def forward(self, data, label=None):
        if label == None:
            return self.__S__.forward(data.reshape(-1, data.size()[-1]).cuda(), aug=False) 
        else:
            data = data.reshape(-1, data.size()[-1]).cuda() 
            outp = self.__S__.forward(data, aug=True)
            outp = outp.reshape(self.num_utt, -1, outp.size()[-1]).transpose(1,0).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp, label)
            return nloss, prec1


class ModelTrainer(object):
    
    def __init__(self, speaker_model, optimizer, scheduler, gpu, **kwargs):
        self.__model__  = speaker_model
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, _ = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler() 
        self.gpu = gpu
        self.ngpu = int(torch.cuda.device_count()) if kwargs.pop('distributed') else 1 # DDP or Not
        self.ndistfactor = int(kwargs.pop('num_utt') * self.ngpu)
        self.swa = kwargs.pop('swa')
        self.swa_start = int(kwargs.pop('swa_start'))
        self.swa_lr = kwargs.pop('swa_lr')
        self.swa_an = kwargs.pop('swa_an')
        self.lr_t0 = kwargs.pop('lr_t0')
        if self.swa:
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.1 * averaged_model_parameter + 0.9 * model_parameter
            self.__swa_model__ = AveragedModel(self.__model__, avg_fn=ema_avg)
            self.__swa_scheduler__ = SWALR(self.__optimizer__, anneal_strategy="linear", anneal_epochs=self.swa_an, swa_lr=self.swa_lr)

    def train_network(self, loader, epoch, verbose):
        self.__model__.train()
        if epoch==1:
            self.__scheduler__.step(0)
        bs = loader.batch_size
        df = self.ndistfactor
        cnt, idx, loss, top1 = 0, 0, 0, 0
        tstart = time.time()
        
        for data, data_label in loader:
            self.__model__.zero_grad()
            data = data.transpose(1,0) 
            label = torch.LongTensor(data_label).cuda()
            with autocast():
                nloss, prec1 = self.__model__(data, label)

            self.scaler.scale(nloss).backward()
            self.scaler.step(self.__optimizer__)
            self.scaler.update()

            loss += nloss.detach().cpu().item()
            top1 += prec1.detach().cpu().item()
            cnt += 1
            idx += bs
            lr = self.__optimizer__.param_groups[0]['lr']
            telapsed = time.time() - tstart
            tstart = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}: Loss {:f}, ACC {:2.3f}%, LR {:.8f} - {:.2f} Hz ".format(idx*df, loader.__len__()*bs*df, loss/cnt, top1/cnt, lr, bs*df/telapsed))
                sys.stdout.flush()

        if self.swa and epoch >= self.swa_start:
            self.__swa_model__.update_parameters(self.__model__)
            self.__swa_scheduler__.step()
        else:
            self.__scheduler__.step()
        return (loss/cnt, top1/cnt, lr)

    def evaluateFromList_with_snorm(self, epoch, test_list, test_path, train_list, train_path, score_norm, tta, num_thread, distributed, top_coh_size, eval_frames=0, num_eval=1, **kwargs):
        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        if self.swa and epoch >= self.swa_start:
            self.__swa_model__.eval()
        else:
            self.__model__.eval()

        ## Eval loader ##
        feats_eval = {}
        tstart = time.time()
        with open(test_list) as f:
            lines_eval = f.readlines()
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines_eval]))
        setfiles = list(set(files))
        setfiles.sort()
        test_dataset = test_dataset_loader(setfiles, test_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
        ds = test_loader.__len__()
        gs = self.ngpu
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                if self.swa and epoch >= self.swa_start:
                    ref_feat = self.__swa_model__(inp1).detach().cpu()
                else:
                    ref_feat = self.__model__(inp1).detach().cpu()
            feats_eval[data[1][0]] = ref_feat
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
                sys.stdout.flush()

        ## Cohort loader if using score normalization ##
        if score_norm:
            feats_coh = {}
            tstart = time.time()
            with open(train_list) as f:
                lines_coh = f.readlines()
            setfiles = list(set([x.split()[0] for x in lines_coh]))
            setfiles.sort()
            cohort_dataset = test_dataset_loader(setfiles, train_path, eval_frames=0, num_eval=1, **kwargs)
            if distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(cohort_dataset, shuffle=False)
            else:
                sampler = None
            cohort_loader = torch.utils.data.DataLoader(cohort_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=sampler)
            ds = cohort_loader.__len__()
            for idx, data in enumerate(cohort_loader):
                inp1 = data[0][0].cuda()
                with torch.no_grad():
                    if self.swa and epoch >= self.swa_start:
                        ref_feat = self.__swa_model__(inp1).detach().cpu()
                    else:
                        ref_feat = self.__model__(inp1).detach().cpu()
                feats_coh[data[1][0]] = ref_feat
                telapsed = time.time() - tstart
                if rank == 0:
                    if idx==0: print('')
                    sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx*gs, ds*gs, idx*gs/telapsed,ref_feat.size()[1]))
                    sys.stdout.flush()
            coh_feat = torch.stack(list(feats_coh.values())).squeeze(1).cuda()
            if self.__model__.module.__L__.test_normalize:
                coh_feat = F.normalize(coh_feat, p=2, dim=1)

        ## Compute verification scores ##
        all_scores, all_labels = [], []
        if distributed:
            ## Gather features from all GPUs
            feats_eval_all = [None for _ in range(0,torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_eval_all, feats_eval)
            if score_norm:
                feats_coh_all = [None for _ in range(0,torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(feats_coh_all, feats_coh)
        if rank == 0:
            tstart = time.time()
            print('')
            ## Combine gathered features
            if distributed:
                feats_eval = feats_eval_all[0]
                for feats_batch in feats_eval_all[1:]:
                    feats_eval.update(feats_batch)
                if score_norm:
                    feats_coh = feats_coh_all[0]
                    for feats_batch in feats_coh_all[1:]:
                        feats_coh.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                data = line.split()
                enr_feat = feats_eval[data[1]].cuda()
                tst_feat = feats_eval[data[2]].cuda()
                if self.__model__.module.__L__.test_normalize:
                    enr_feat = F.normalize(enr_feat, p=2, dim=1)
                    tst_feat = F.normalize(tst_feat, p=2, dim=1)

                if tta==True and score_norm==True:
                    print('Not considered condition')
                    exit()
                if tta == False:
                    score = F.cosine_similarity(enr_feat, tst_feat)

                if score_norm:
                    score_e_c = F.cosine_similarity(enr_feat, coh_feat)
                    score_c_t = F.cosine_similarity(coh_feat, tst_feat)

                    if top_coh_size == 0: top_coh_size = len(coh_feat)
                    score_e_c = torch.topk(score_e_c, k=top_coh_size, dim=0)[0]
                    score_c_t = torch.topk(score_c_t, k=top_coh_size, dim=0)[0]
                    score_e = (score - torch.mean(score_e_c, dim=0)) / torch.std(score_e_c, dim=0)
                    score_t = (score - torch.mean(score_c_t, dim=0)) / torch.std(score_c_t, dim=0)
                    score = 0.5 * (score_e + score_t)

                elif tta:
                    score = torch.mean(F.cosine_similarity(enr_feat.unsqueeze(-1), tst_feat.unsqueeze(-1).transpose(0,2)))

                all_scores.append(score.detach().cpu().numpy())
                all_labels.append(int(data[0]))
                telapsed = time.time() - tstart
                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines_eval), idx/telapsed))
                sys.stdout.flush()
        return (all_scores, all_labels)

    def saveParameters(self, path):
        torch.save(self.__model__.module.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d"%self.gpu)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("{} is not in the model.".format(origname))
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)
