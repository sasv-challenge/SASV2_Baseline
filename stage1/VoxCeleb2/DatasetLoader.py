#! /usr/bin/python
# -*- encoding: utf-8 -*-
import os
import glob
import torch
import random
import soundfile
import numpy as np
import torch.distributed as dist
from scipy import signal
from torch.utils.data import Dataset
from utils import Resample

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    # Maximum audio length
    max_audio = max_frames * 160 #+ 240
    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1 
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]
    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    feats = []
    if evalmode and max_frames == 0:
        feats += [audio]
    else:
        for asf in startframe:
            feats += [audio[int(asf):int(asf)+max_audio]]
    feat = np.stack(feats,axis=0).astype(np.float)
    return feat

class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_audio = max_frames * 160 #+ 240
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]] += [file]
        self.rir_files = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        self.perturb_prob = 1.0
        self.speeds = [95, 105] 
        self.sample_rate = 16000
        self.resamplers = []
        for speed in self.speeds:
            config = {
                "orig_freq": self.sample_rate,
                "new_freq" : self.sample_rate*speed//100,
            }
            self.resamplers += [Resample(**config)]

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio**2) + 1e-4) 
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0]**2) + 1e-4) 
            noises += [np.sqrt(10**((clean_db - noise_db - noise_snr) / 10)) * noiseaudio]
        return np.sum(np.concatenate(noises,axis=0), axis=0,keepdims=True) + audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, fs = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

    def speed_perturb(self, audio):
        if torch.rand(1) > self.perturb_prob:
            return audio
        samp_index = random.randint(0, len(self.speeds)-1)
        return self.resamplers[samp_index](torch.FloatTensor(audio)).detach().cpu().numpy()


class train_dataset_loader(Dataset):
    
    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path, **kwargs):
        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)
        self.train_list = train_list
        self.max_frames = max_frames
        self.max_audio = max_frames*160 #+ 240
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment = augment

        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[1] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_list = []
        self.data_label = []
        for lidx, line in enumerate(lines):
            data = line.strip().split()
            speaker_label = dictkeys[data[1]]
            filename = os.path.join(train_path, data[0])
            self.data_label += [speaker_label]
            self.data_list += [filename]

    def __getitem__(self, indices):
        feat = []
        for index in indices:
            audio = loadWAV(self.data_list[index], self.max_frames, evalmode=False)
            if self.augment:
                augtype = random.randint(0,6)
                if augtype == 1:
                    audio = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio = self.augment_wav.additive_noise('music', audio)
                elif augtype == 3:
                    audio = self.augment_wav.additive_noise('speech', audio)
                elif augtype == 4:
                    audio = self.augment_wav.additive_noise('noise', audio)
                elif augtype == 5:
                    audio = self.augment_wav.additive_noise('speech', audio)
                    audio = self.augment_wav.additive_noise('music', audio)
                elif augtype == 6:
                    audio = self.augment_wav.speed_perturb(audio)
                    if audio.shape[1] > self.max_audio:
                        audio = audio[:, 0 : self.max_audio]
                    else:
                        audio = np.pad(audio[0], (0, self.max_audio-audio.shape[1]), 'wrap')
                        audio = np.expand_dims(audio, 0)
            feat += [audio]
        feat = np.concatenate(feat, axis=0)
        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class test_dataset_loader(Dataset):
    
    def __init__(self, test_list, test_path, eval_frames, num_eval, label=False, **kwargs):
        self.max_frames = eval_frames
        self.num_eval = num_eval
        self.test_path = test_path
        self.test_list = test_list
        self.test_label = label
        
    def __getitem__(self, index):
        audio = loadWAV(os.path.join(self.test_path, self.test_list[index]), self.max_frames, evalmode=True, num_eval=self.num_eval)
        if self.test_label!=False: 
            return torch.FloatTensor(audio), self.test_list[index], self.test_label[index]
        else:
            return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


class train_dataset_sampler(torch.utils.data.Sampler):
    
    def __init__(self, data_source, num_utt, max_seg_per_spk, num_spk, distributed, seed, **kwargs):
        self.data_label = data_source.data_label
        self.num_utt = num_utt
        self.max_seg_per_spk = max_seg_per_spk
        self.num_spk = num_spk
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed
        self.batch_size = num_utt * num_spk
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()
        data_dict = {}
        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label] += [index]

        ## Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()
        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
        flattened_list = []
        flattened_label = []
        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(min(len(data),self.max_seg_per_spk),self.num_utt)
            rp = lol(np.arange(numSeg),self.num_utt)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list += [[data[i] for i in indices]]

        ## Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        ## Reduce data waste referred from https://github.com/clovaai/voxceleb_trainer/pull/136/files
        resmixid = []
        mixlabel_ins = 1

        if self.num_utt != 1:
            while len(mixid) > 0 and mixlabel_ins > 0:
                mixlabel_ins = 0
                for ii in mixid:
                    startbatch = round_down(len(mixlabel), self.num_spk)
                    if flattened_label[ii] not in mixlabel[startbatch:]:
                        mixlabel += [flattened_label[ii]]
                        mixmap += [ii]
                        mixlabel_ins += 1
                    else:
                        resmixid += [ii]
                mixid = resmixid
                resmixid = []
        else:
            for ii in mixid:
                startbatch = round_down(len(mixlabel), self.num_spk)
                mixlabel += [flattened_label[ii]]
                mixmap += [ii]
        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size = round_down(len(mixed_list), self.num_spk * dist.get_world_size()) 
            start_index = int ((dist.get_rank()) / dist.get_world_size() * total_size)
            end_index = int ((dist.get_rank() + 1) / dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.num_spk)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])
    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
