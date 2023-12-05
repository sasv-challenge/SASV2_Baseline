#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys
import time
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler


class WrappedModel(nn.Module):
    
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SASVNet(nn.Module):
    
    def __init__(self, model, trainfunc, num_utt, **kwargs):
        super(SASVNet, self).__init__()
        SASVNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SASVNetModel(**kwargs)
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
    
    def __init__(self, speaker_model, optimizer, scheduler, **kwargs):
        self.__model__  = speaker_model
        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)
        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, _ = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler() 
        self.gpu = 0
        self.ngpu = 1
        self.ndistfactor = int(kwargs.get('num_utt') * self.ngpu)

    def train_network(self, loader, epoch):
        self.__model__.train()
        self.__scheduler__.step(epoch-1)
        
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

            sys.stdout.write("\rProcessing {:d} of {:d}: Loss {:f}, ACC {:2.3f}%, LR {:.8f} - {:.2f} Hz  ".format(idx*df, loader.__len__()*bs*df, loss/cnt, top1/cnt, lr, bs*df/telapsed))
            sys.stdout.flush()

        return (loss/cnt, top1/cnt, lr)

    def evaluateFromList(self, eval_list, eval_path, num_thread, eval_frames=0, num_eval=1, **kwargs):

        rank = 0
        self.__model__.eval()

        ## Enroll (speaker model) loader ##
        spk_meta = {}
        meta_f = np.loadtxt('protocols/ASVspoof2019.LA.asv.eval.female.trn.txt', str)
        meta_m = np.loadtxt('protocols/ASVspoof2019.LA.asv.eval.male.trn.txt', str)
        meta = np.concatenate((meta_f, meta_m))
        for i, spk in enumerate(meta[:,0]):
            spk_meta[spk] = meta[i][1].split(',')
        
        embeds_enr = {}
        files = []
        for idx1, spk in enumerate(spk_meta):
            for file in spk_meta[spk]:
                files += [file + '.flac']

            test_dataset = test_dataset_loader(files, eval_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=None)
            ref_embeds = []
            for _, data in enumerate(test_loader):
                inp1 = data[0][0].cuda()
                with torch.no_grad():
                    ref_embed = self.__model__(inp1).detach().cpu()
                ref_embeds += [ref_embed]
            embeds_enr[spk] = ref_embeds
            # embeds_enr[spk] = torch.mean(torch.stack(ref_embeds), dim=0)
            files = []
            if rank == 0:
                sys.stdout.write("\r Enrollment bona-fide speaker model: {:s}, {:d} of {:d}      ".format(spk, idx1, len(spk_meta.keys())))
                sys.stdout.flush()
        print('')

        ## Test loader ##
        tstart = time.time()
        with open(eval_list) as f:
            lines_eval = f.readlines()
        files = [x.strip().split(' ')[1] + '.flac' for x in lines_eval]
        setfiles = list(set(files))
        setfiles.sort()

        test_dataset = test_dataset_loader(setfiles, eval_path, eval_frames=eval_frames, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_thread, drop_last=False, sampler=None)

        ds = test_loader.__len__()
        gs = self.ngpu
        
        embeds_tst = {}
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_embed = self.__model__(inp1).detach().cpu()
            embeds_tst[data[1][0][:-5]] = ref_embed
            telapsed = time.time() - tstart
            if rank == 0:
                sys.stdout.write("\r Reading {:d} of {:d}: {:.2f} Hz, embedding size {:d}      ".format(idx*gs, ds*gs, idx*gs/telapsed, ref_embed.size()[1]))
                sys.stdout.flush()

        ## Compute verification scores ##
        all_scores, all_labels = [], []
        if rank == 0:
            tstart = time.time()
            print('')
            for key in embeds_enr.keys():
                embeds_enr[key] = torch.mean(torch.stack(embeds_enr[key]), dim=0)


            ## Read files and compute all scores
            for idx, line in enumerate(lines_eval):
                data = line.split()
                enr = embeds_enr[data[0]].cuda()
                tst = embeds_tst[data[1]].cuda()
                if self.__model__.module.__L__.test_normalize:
                    enr = F.normalize(enr, p=2, dim=1)
                    tst = F.normalize(tst, p=2, dim=1)

                score = F.cosine_similarity(enr, tst)

                all_scores.append(score.detach().cpu().numpy())
                all_labels.append(data[3])

                telapsed = time.time() - tstart

                sys.stdout.write("\r Computing {:d} of {:d}: {:.2f} Hz      ".format(idx, len(lines_eval), idx/telapsed))
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
