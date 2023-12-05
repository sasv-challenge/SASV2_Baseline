#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import sys
import time
import glob
import torch
import zipfile
import warnings
import argparse
import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from metrics import *
from SASVNet import *
from DatasetLoader import *
from tuneThreshold import *

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description = "SASVNet")
## Data loader
parser.add_argument('--max_frames',     type=int,   default=500,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=0,      help='Input length to the network for testing. 0 uses the whole files')
parser.add_argument('--num_eval',       type=int,   default=1,      help='Number of segments of input utterence for testing')
parser.add_argument('--num_spk',        type=int,   default=40,     help='Number of non-overlapped bona-fide speakers within a batch')
parser.add_argument('--num_utt',        type=int,   default=2,      help='Number of utterances per speaker within a batch')
parser.add_argument('--batch_size',     type=int,   default=160,    help='batch_size = num_spk*num_utt + num_spf, num_spf = batch_size - num_spk*num_utt')
parser.add_argument('--max_seg_per_spk',type=int,   default=10000,  help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--num_thread',     type=int,   default=10,     help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,      help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=100,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="aamsoftmax",     help='Loss function')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam",     help='sgd, adam, adamW, or adamP')
parser.add_argument('--scheduler',      type=str,   default="cosine_annealing_warmup_restarts",     help='Learning rate scheduler')
parser.add_argument('--weight_decay',   type=float, default=1e-7,   help='Weight decay in the optimizer')
parser.add_argument('--lr',             type=float, default=1e-4,   help='Initial learning rate')
parser.add_argument('--lr_t0',          type=int,   default=8,      help='Cosine sched: First cycle step size')
parser.add_argument('--lr_tmul',        type=float, default=1.0,    help='Cosine sched: Cycle steps magnification.')
parser.add_argument('--lr_max',         type=float, default=1e-4,   help='Cosine sched: First cycle max learning rate')
parser.add_argument('--lr_min',         type=float, default=0,      help='Cosine sched: First cycle min learning rate')
parser.add_argument('--lr_wstep',       type=int,   default=0,      help='Cosine sched: Linear warmup step size')
parser.add_argument('--lr_gamma',       type=float, default=0.8,    help='Cosine sched: Decrease rate of max learning rate by cycle')

## Loss functions
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--num_class',      type=int,   default=41,     help='Number of speakers in the softmax layer, 40 or 20 (speaker-classes) + 1 (spoofing-class)') # 41

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="./exp",     help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="",     help='Train list')
parser.add_argument('--eval_list',      type=str,   default="",     help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="",     help='Absolute path to the train set')
parser.add_argument('--eval_path',      type=str,   default="",     help='Absolute path to the test set')
parser.add_argument('--spk_meta_train', type=str,   default="",     help='')
parser.add_argument('--spk_meta_eval',  type=str,   default="",     help='')
parser.add_argument('--musan_path',     type=str,   default="",     help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="",     help='Absolute path to the test set')

## Model definition
parser.add_argument('--num_mels',       type=int,   default=80,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=True,   help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition')
parser.add_argument('--pooling_type',   type=str,   default="ASP",  help='Type of encoder')
parser.add_argument('--num_out',        type=int,   default=192,    help='Embedding size in the last FC layer')
parser.add_argument('--eca_c',          type=int,   default=1024,   help='ECAPA-TDNN channel')
parser.add_argument('--eca_s',          type=int,   default=8,      help='ECAPA-TDNN model-scale')

## Evaluation types
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
parser.add_argument('--scoring',        dest='scoring', action='store_true', help='Scoring')

args = parser.parse_args()

def main_worker(args):
    # args.gpu = gpu

    ## Load models
    s = SASVNet(**vars(args))
    s = WrappedModel(s).cuda()

    it = 1
    SASV_EERs, SV_EERs, SPF_EERs = [], [], []
    ## Write args to scorefile
    scorefile   = open(args.result_save_path+"/scores.txt", "a+")
    ## Print params
    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())
    print('Total parameters: {:.2f}M'.format(float(pytorch_total_params)/1024/1024))

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

    ## Evaluation only
    if args.eval == True:
        print('Test list',args.eval_list)
        sc, lab = trainer.evaluateFromList(**vars(args))

        ## Save scores
        if args.scoring == True:
            np.save(f'{args.result_save_path}/predict_scores.npy', sc)
            np.save(f'{args.result_save_path}/ground_truth.npy', lab)

        sasv_eer, sv_eer, spf_eer = get_all_EERs(sc, lab)
        msg = f"SASV-EER {sasv_eer:2.4f}, SV-EER {sv_eer:2.4f}, SPF-EER {spf_eer:2.4f}"
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S")
        print('\n', cur_time, msg)
        with open(args.result_save_path + "/metrics", "a") as f_res:
            f_res.write(cur_time + "\n")
            f_res.write(msg)


        return

    ## Initialise trainer and data loader
    train_dataset = train_dataset_loader(**vars(args))
    train_sampler = train_dataset_sampler(train_dataset, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size//2, #args.num_spk,
        num_workers=args.num_thread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    ## Update learning rate
    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Save training code and params
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
        loss, traineer, lr = trainer.train_network(train_loader, it)
        print('')

        ## Evaluating
        if it % args.test_interval == 0:
            sc, lab = trainer.evaluateFromList(epoch=it, **vars(args))

            sasv_eer, sv_eer, spf_eer = get_all_EERs(sc, lab)
            SASV_EERs += [sasv_eer]
            SV_EERs += [sv_eer]
            SPF_EERs += [spf_eer]

            print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "Epoch {:d}, ACC {:2.2f}, TLOSS {:f}, LR {:2.8f}, SASV_EER {:2.4f}, SV_EER {:2.4f}, SPF_EER {:2.4f}, BestSASV_EER {:2.4f}, BestSV_EER {:2.4f}, BestSPF_EER {:2.4f}".format(it, traineer, loss, lr, sasv_eer, sv_eer, spf_eer, min(SASV_EERs), min(SV_EERs), min(SPF_EERs)))
            scorefile.write("Epoch {:d}, ACC {:2.2f}, TLOSS {:f}, LR {:2.8f}, SASV_EER {:2.4f}, SV_EER {:2.4f}, SPF_EER {:2.4f}, BestSASV_EER {:2.4f}, BestSV_EER {:2.4f}, BestSPF_EER {:2.4f}\n".format(it, traineer, loss, lr, sasv_eer, sv_eer, spf_eer, min(SASV_EERs), min(SV_EERs), min(SPF_EERs)))
            scorefile.flush()
            trainer.saveParameters(args.model_save_path+"/model%09d.model"%it)
            print('')

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

    main_worker(args)

if __name__ == '__main__':
    main()
