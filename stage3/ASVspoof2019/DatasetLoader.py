#! /usr/bin/python
# -*- encoding: utf-8 -*-
import os
import glob
import torch
import random
import itertools
import soundfile
import numpy as np
# import torch.distributed as dist
from scipy import signal
from torch.utils.data import Dataset
from utils import Resample

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def loadWAV(filename, max_frames, evalmode=True, num_eval=1):
    
    # Maximum audio length
    max_audio = max_frames * 160
    
    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)
    audiosize = audio.shape[0]
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1 
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]
    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
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
        self.max_audio = max_frames * 160
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise':[0, 15], 'speech':[13, 20], 'music':[5, 15]}
        self.numnoise = {'noise':[1, 1], 'speech':[3, 7], 'music':[1, 1]}
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
        return signal.convolve(audio, rir, mode='full')[:, : self.max_audio]

    def speed_perturb(self, audio):
        if torch.rand(1) > self.perturb_prob:
            return audio
        samp_index = random.randint(0, len(self.speeds)-1)
        return self.resamplers[samp_index](torch.FloatTensor(audio)).detach().cpu().numpy()


class train_dataset_loader(Dataset):
    
    def __init__(self, train_list, augment, musan_path, rir_path, max_frames, train_path, **kwargs):
        self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames=max_frames)
        self.train_list = train_list
        self.max_frames = max_frames
        self.max_audio = max_frames * 160
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment = augment

        # Read training files
        with open(train_list) as dataset_file:
            lines = dataset_file.readlines()

        # Make a dictionary of ID names and ID indices
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys += ['spoof']
        dictkeys.sort()
        dictkeys = {key : ii for ii, key in enumerate(dictkeys)}

        # Parse the training list into file names and ID indices
        self.data_list = []        

        self.data_label = []
        self.data_group = [] # 'speaker_type' (e.g., 'LA0039_A01')
        
        for idx, line in enumerate(lines):
            data = line.strip().split()
            filename = os.path.join(train_path, data[1])
            self.data_list += [filename + '.wav']          
            
            if data[4] == 'bonafide':
                self.data_label += [dictkeys[data[0]]]
            else:
                self.data_label += [dictkeys['spoof']]
                
            self.data_group += [data[0] + '_' + data[3]]
            
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
    
    def __init__(self, data_source, num_utt, max_seg_per_spk, num_spk, batch_size, seed, **kwargs):       
        self.data_group = data_source.data_group
        self.num_utt = num_utt
        self.max_seg_per_spk = max_seg_per_spk
        self.num_spk = num_spk
        self.epoch = 0
        self.seed = seed
        # self.distributed = distributed
        self.batch_size = batch_size
        self.num_spoof = batch_size//2 - num_spk
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_group), generator=g).tolist()
        data_dict = {}
        
        # Sort into dictionary of file indices for each ID
        for index in indices:
            typ = self.data_group[index][8:]
            if typ == '-':
                typ = 'bonafide'
            else:
                typ = 'spoof'
            spk = self.data_group[index][:7]
            if not typ in data_dict:
                data_dict[typ] = {}
            if not spk in data_dict[typ]:
                data_dict[typ][spk] = []
            data_dict[typ][spk] += [index]

        ## Group file indices for each class
        dictkeys_typ = list(data_dict.keys())
        dictkeys_typ.sort()
        dictkeys_spk = [list(data_dict[t].keys()) for t in dictkeys_typ]
        dictkeys_spk = list(set(list(itertools.chain(*dictkeys_spk))))
        dictkeys_spk.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
        
        flattened_lst = []
        flattened_typ = []
        flattened_spk = []
        for _, typ_key in enumerate(dictkeys_typ):
            for _, spk_key in enumerate(dictkeys_spk):
                if spk_key in data_dict[typ_key].keys():
                    data = data_dict[typ_key][spk_key]
                    numSeg = round_down(min(len(data), self.max_seg_per_spk), self.num_utt)
                    rp = lol(np.arange(numSeg), self.num_utt)
                    flattened_typ.extend([typ_key] * (len(rp)))
                    flattened_spk.extend([spk_key] * (len(rp)))
                else:
                    pass
                for indices in rp:
                    flattened_lst += [[data[i] for i in indices]]
        
        flattened_lst = np.array(flattened_lst)
        flattened_typ = np.array(flattened_typ)
        flattened_spk = np.array(flattened_spk)
        
        flattened_lsts = {}
        flattened_spks = {}
        for typ in dictkeys_typ:
            if typ not in flattened_lsts.keys():
                flattened_lsts[typ] = []
                flattened_spks[typ] = []
            idx = np.where(flattened_typ==typ)[0]
            flattened_lsts[typ] += flattened_lst[idx].tolist()
            flattened_spks[typ] += flattened_spk[idx].tolist()

        ## Mix data in random order
        mixid = {}
        for typ in dictkeys_typ:
            if typ not in mixid.keys():
                mixid[typ] = []
            mixid[typ] += torch.randperm(len(flattened_lsts[typ]), generator=g).tolist()
        
        mixspks = {}
        mixmaps = {}
        for typ in dictkeys_typ:
            # Bona-fide data: non-overlapped speakers with num_utt (i.e., 2) utterances
            if typ == 'bonafide':
                resmixid = []
                mixlab_idx = 1
                if typ not in mixspks.keys():
                    mixspks[typ] = []
                    mixmaps[typ] = []
                while len(mixid[typ]) > 0 and mixlab_idx > 0:
                    mixlab_idx = 0
                    for ii in mixid[typ]:
                        startbatch = round_down(len(mixspks[typ]), self.num_spk)
                        if flattened_spks[typ][ii] not in mixspks[typ][startbatch:]:
                            mixspks[typ] += [flattened_spks[typ][ii]]
                            mixmaps[typ] += [ii]
                            mixlab_idx += 1
                        else:
                            resmixid += [ii]
                    mixid[typ] = resmixid
                    resmixid = []

            # Spoofing data: random sampling
            else:
                mixspks[typ] = []
                mixmaps[typ] = []
                for ii in mixid[typ]:
                    mixspks[typ] += [flattened_spks[typ][ii]]
                    mixmaps[typ] += [ii]

        mixed_lists = {}
        for typ in dictkeys_typ:
            if typ not in mixspks.keys():
                mixed_lists[typ] = []
            mixed_lists[typ] = [flattened_lsts[typ][i] for i in mixmaps[typ]]
        
        # Organizing batch configuration: [bona-fide data | spoofing data]
        mixed_list = []
        num_iter = min(len(mixed_lists['bonafide'])//self.num_spk, len(mixed_lists['spoof'])//self.num_spoof)
        for i in range(num_iter):
            batch_list = mixed_lists['bonafide'][i*self.num_spk: (i+1)*self.num_spk] + mixed_lists['spoof'][i*self.num_spoof: (i+1)*self.num_spoof]
            mixed_list.extend(batch_list)
        
        total_size = round_down(len(mixed_list), self.num_spk)
        self.num_samples = total_size
        return iter(mixed_list[:total_size])
    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
