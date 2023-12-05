#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import sys
import time
import glob
import torch
import socket
import zipfile
import warnings
import argparse
import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from SpeakerNet import *
from DatasetLoader import *
from tuneThreshold import *

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "SpeakerNet")
## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=0,      help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--num_eval',       type=int,   default=1,      help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--num_spk',        type=int,   default=100,    help='Number of speakers per batch, i.e., batch size = num_spk * num_utt')
parser.add_argument('--num_utt',        type=int,   default=2,      help='Number of utterances per speaker in batch')
parser.add_argument('--max_seg_per_spk',type=int,   default=600,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--num_thread',     type=int,   default=10,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=True,   help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,      help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="aamsoftmaxproto", help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adamW", help='sgd, adam, adamW, or adamP')
parser.add_argument('--scheduler',      type=str,   default="cosine_annealing_warmup_restarts", help='Learning rate scheduler [cosine_annealing_warmup_restarts, steplr, or cycliclr]')
parser.add_argument('--weight_decay',   type=float, default=1e-7,   help='Weight decay in the optimizer')
parser.add_argument('--lr',             type=float, default=0.001,  help='StepLR sched: Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.50,   help='StepLR sched: Learning rate decay')
parser.add_argument("--lr_decay_interval",type=int, default=4,      help='StepLR sched: Learning rate decay every [lr_interval] epochs')
parser.add_argument('--lr_t0',          type=int,   default=25,     help='Cosine sched: First cycle step size') # 3gpu: 6160
parser.add_argument('--lr_tmul',        type=float, default=1.0,    help='Cosine sched: Cycle steps magnification.')
parser.add_argument('--lr_max',         type=float, default=1e-3,   help='Cosine sched: First cycle max learning rate')
parser.add_argument('--lr_min',         type=float, default=1e-8,   help='Cosine sched: First cycle min learning rate')
parser.add_argument('--lr_wstep',       type=int,   default=10,     help='Cosine sched: Linear warmup step size')
parser.add_argument('--lr_gamma',       type=float, default=0.5,    help='Cosine sched: Decrease rate of max learning rate by cycle')
parser.add_argument("--lr_cyclic_min",  type=float, default=1e-8,   help='CyclicLR sched: Minimun learning rate')
parser.add_argument("--lr_cyclic_max",  type=float, default=1e-3,   help='CyclicLR sched: Maximun learning rate')
parser.add_argument('--lr_up_size',     type=int,   default=10,     help='CyclicLR sched: Up cycle size')
parser.add_argument('--lr_down_size',   type=int,   default=15,     help='CyclicLR sched: Down cycle size')
parser.add_argument('--lr_mode',        type=str,   default='',     help='CyclicLR sched: Mode: triangular, triangular2, or exp_range')
parser.add_argument('--swa',            dest='swa', action='store_true', help='Stochastic weight average')
parser.add_argument('--swa_start',      type=int,   default=26,     help='Stochastic weight average starting point')
parser.add_argument('--swa_lr',         type=float, default=1e-6,   help='Stochastic weight average ')
parser.add_argument('--swa_an',         type=int,   default=10,     help='Stochastic weight average annealing epoch')

## Loss functions
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--num_class',      type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--w_cls',          type=float, default=1.0,    help='Weight for softmax-based loss')
parser.add_argument('--w_mtr',          type=float, default=1.0,    help='Weight for metric-based loss')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="",     help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="./list/train_vox2.txt", help='Train list')
parser.add_argument('--test_list',      type=str,   default="./list/veri_test2.txt", help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="",     help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="/home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="/home/shmun/DB/MUSAN/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="/home/shmun/DB/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')

## Model definition
parser.add_argument('--num_mels',       type=int,   default=80,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=True,   help='Log input features')
parser.add_argument('--model',          type=str,   default="SKA_TDNN", help='Name of model definition')
parser.add_argument('--pooling_type',   type=str,   default="ASP",  help='Type of encoder')
parser.add_argument('--num_out',        type=int,   default=192,    help='Embedding size in the last FC layer')
parser.add_argument('--eca_c',          type=int,   default=1024,   help='ECAPA-TDNN channel')
parser.add_argument('--eca_s',          type=int,   default=8,      help='ECAPA-TDNN model-scale')

## Evaluation types
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
parser.add_argument('--score_norm',     dest='score_norm', action='store_true', help='Score normalization')
parser.add_argument('--type_coh',       type=str,   default='utt',  help='Cohort type - select: spk or utt')
parser.add_argument('--top_coh_size',   type=int,   default=20000,  help='Maximum cohort size for adaptive s-norm')
parser.add_argument('--tta',            dest='tta', action='store_true', help='Test Time Augmentation')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8000", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
args = parser.parse_args()

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ## Load models
    s = SpeakerNet(**vars(args))
    if args.distributed:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']=args.port
        dist.init_process_group(backend='nccl', world_size=ngpus_per_node, rank=args.gpu)
        torch.cuda.set_device(args.gpu)
        s.cuda(args.gpu)
        s = torch.nn.parallel.DistributedDataParallel(s, device_ids=[args.gpu], find_unused_parameters=True)
        print('Loaded the model on GPU {:d}'.format(args.gpu))
    else:
        s = WrappedModel(s).cuda(args.gpu)
    it = 1
    EERs, DCFs  = [], []
    if args.gpu == 0:
        ## Write args to scorefile
        scorefile   = open(args.result_save_path+"/scores.txt", "a+")
        ## Print params
        pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())
        print('Total parameters: {:.2f}M'.format(float(pytorch_total_params)/1024/1024))

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))
    train_sampler = train_dataset_sampler(train_dataset, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.num_spk,
        num_workers=args.num_thread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )
    trainer = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('%s/model0*.model'%args.model_save_path)
    modelfiles.sort()
    if(args.initial_model != ""):
        trainer.loadParameters(args.initial_model)
        print("Model {} loaded!".format(args.initial_model))
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1])
        print("Model {} loaded from previous state!".format(modelfiles[-1]))
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    
    ## Update learning rate
    for ii in range(1,it):
        trainer.__scheduler__.step()
    
    ## Evaluation code - must run on single GPU
    if args.eval == True:
        print('Test list',args.test_list)
        sc, lab = trainer.evaluateFromList_with_snorm(epoch=it, **vars(args))
        if args.gpu == 0:
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])
            fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
            mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}".format(result[1]),"MinDCF {:2.5f}".format(mindcf))
        return

    ## Save training code and params
    if args.gpu == 0:
        pyfiles = glob.glob('./*.py')
        strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        zipf = zipfile.ZipFile(args.result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
        for file in pyfiles:
            zipf.write(file)
        zipf.close()
        with open(args.result_save_path + '/run%s.cmd'%strtime, 'w') as f:
            f.write('%s'%args)

    ## Core training script
    for it in range(it,args.max_epoch+1):
        ## Training
        train_sampler.set_epoch(it)
        loss, traineer, lr = trainer.train_network(train_loader, it, verbose=(args.gpu == 0))
        if args.gpu == 0:
            print('')

        ## Evaluating
        if it % args.test_interval == 0:
            #if args.swa and it % args.lr_t0 >= args.swa_start:
            if args.swa and it >= args.swa_start:
                torch.optim.swa_utils.update_bn(train_loader, trainer.__swa_model__)
            sc, lab = trainer.evaluateFromList_with_snorm(epoch=it, **vars(args))
            if args.gpu == 0:
                result = tuneThresholdfromScore(sc, lab, [1, 0.1])
                fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
                mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)
                EERs += [result[1]]
                DCFs += [mindcf]
                print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, ACC {:2.2f}, TLOSS {:f}, LR {:2.8f}, EER {:2.4f}, MinDCF {:2.5f}, bestEER {:2.4f}, bestMinDCF {:2.5f}".format(it, traineer, loss, lr, result[1], mindcf, min(EERs), min(DCFs)))
                scorefile.write("Epoch {:d}, ACC {:2.2f}, TLOSS {:f}, LR {:2.8f}, EER {:2.4f}, MinDCF {:2.5f}, bestEER {:2.4f}, bestMinDCF {:2.5f}\n".format(it, traineer, loss, lr, result[1], mindcf, min(EERs), min(DCFs)))
                trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)
                scorefile.flush()
                print('')
    if args.gpu == 0:
        scorefile.close()

def main():
    args.model_save_path  = args.save_path+"/model"
    args.result_save_path = args.save_path+"/result"
    if os.path.exists(args.model_save_path): print("[Folder {} already exists...]".format(args.save_path))
    os.makedirs(args.model_save_path, exist_ok=True)
    os.makedirs(args.result_save_path, exist_ok=True)
    n_gpus = torch.cuda.device_count()
    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Save path:',args.save_path)
    if args.distributed:
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, args))
    else:
        main_worker(0, None, args)

if __name__ == '__main__':
    main()