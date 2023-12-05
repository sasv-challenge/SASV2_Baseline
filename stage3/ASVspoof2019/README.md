# Stage 3

This repository is developed based on the [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) and [ska-tdnn](https://github.com/msh9184/ska-tdnn).

## Dependencies
If you use the Anaconda virtual environment,
```
conda create -n sasv python=3.9 cudatoolkit=11.3
conda activate sasv
```
Install all dependency packages,
```
pip3 install -r requirements.txt
```

## Models
Three models are included in this repository. You can select the model by the `--model` option:
```
ECAPA_TDNN [1]
MFA_Conformer [2]
SKA_TDNN [3]
```

[1] B. Desplanques, J. Thienpondt, and K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," in *Proc. INTERSPEECH*, 2020, pp. 3707-3711.

[2] Y. Zhang, Z. Lv, H. Wu, S. Zhang, P. Hu, Z. Wu, H. Lee, and H. Meng., “MFA-Conformer: Multi-scale Feature Aggregation Conformer for Automatic Speaker Verification,” in *Proc. INTERSPEECH*, 2022.

[3] S. H. Mun, J. Jung, M. H. Han, and N. S. Kim, "Frequency and Multi-Scale Selective Kernel Attention for Speaker Verification," in *Proc. IEEE SLT*, 2022.


## Training
Training example 1: `SKA_TDNN` from scratch using `ASVspoof2019 LA train+dev`,

```
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
        --max_frames 500 \
        --num_spk 40 \
        --num_utt 2 \
        --batch_size 160 \
        --trainfunc sasv_e2e_v1 \
        --optimizer adamW \
        --scheduler cosine_annealing_warmup_restarts \
        --lr_t0 8 \
        --lr_tmul 1.0 \
        --lr_max 1e-4 \
        --lr_min 0 \
        --lr_wstep 0 \
        --lr_gamma 0.8 \
        --margin 0.2 \
        --scale 30 \
        --num_class 41 \
        --save_path ./save/sasv_baseline_stage3 \
        --train_list ./protocols/ASVspoof2019.LA.cm.train_dev.trn.txt \
        --eval_list ./protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt \
        --train_path /path/to/dataset/ASVSpoof/ASVSpoof2019/LA \
        --eval_path /path/to/dataset/ASVSpoof/ASVSpoof2019/LA/ASVspoof2019_LA_eval/flac \
        --spk_meta_train ./spk_meta/spk_meta_trn.pk
        --spk_meta_eval ./spk_meta/spk_meta_eval.pk
        --musan_path /path/to/dataset/MUSAN/musan_split \
        --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \
        --model SKA_TDNN
```

Training example 2: `MFA_Conformer` with pre-trained weight using `ASVspoof2019 LA train`,
```
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
        --max_frames 500 \       
        --num_spk 20 \
        --num_utt 2 \
        --batch_size 80 \
        --trainfunc sasv_e2e_v1 \
        --optimizer adamW \
        --scheduler cosine_annealing_warmup_restarts \
        --lr_t0 8 \
        --lr_tmul 1.0 \
        --lr_max 1e-4 \
        --lr_min 0 \
        --lr_wstep 0 \
        --lr_gamma 0.8 \
        --margin 0.2 \
        --scale 30 \
        --num_class 21 \
        --save_path ./save/sasv_baseline_stage3 \
        --train_list ./protocols/ASVspoof2019.LA.cm.train_dev.trn.txt \
        --eval_list ./protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt \
        --train_path /path/to/dataset/ASVSpoof/ASVSpoof2019/LA \
        --eval_path /path/to/dataset/ASVSpoof/ASVSpoof2019/LA/ASVspoof2019_LA_eval/flac \
        --spk_meta_train ./spk_meta/spk_meta_trn.pk
        --spk_meta_eval ./spk_meta/spk_meta_eval.pk
        --musan_path /path/to/dataset/MUSAN/musan_split \
        --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \
        --model MFA_Conformer \
        --initial_model /path/to/your_model/pretrained_weight.model
```
[In this repository](https://github.com/sasv-challenge/ASVSpoof5-SASVBaseline), you can download several pre-trained weights used in [this paper](https://arxiv.org/pdf/2305.19051.pdf) and fine-tune them using the above command.

## Evaluation
Evaluation example: `SKA_TDNN` using `SASV protocol` on the ASVspoof2019 LA eval,
```
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
        --eval \
        --eval_frames 0 \
        --num_eval 1 \
        --eval_list ./protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt \
        --eval_path /path/to/dataset/ASVSpoof/ASVSpoof2019/LA/ASVspoof2019_LA_eval/flac \
        --model SKA_TDNN \
        --initial_model /path/to/your_model/pretrained_weight.model
```

## Citation
If you utilize this repository, please cite the following paper,
```
@inproceedings{chung2020in,
  title={In defence of metric learning for speaker recognition},
  author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
  booktitle={Proc. Interspeech},
  year={2020}
}
```

```
@inproceedings{jung2022pushing,
  title={Pushing the limits of raw waveform speaker recognition},
  author={Jung, Jee-weon and Kim, You Jin and Heo, Hee-Soo and Lee, Bong-Jin and Kwon, Youngki and Chung, Joon Son},
  booktitle={Proc. Interspeech},
  year={2022}
}
```

```
@inproceedings{mun2022frequency,
  title={Frequency and Multi-Scale Selective Kernel Attention for Speaker Verification},
  author={Mun, Sung Hwan and Jung, Jee-weon and Han, Min Hyun and Kim, Nam Soo},
  booktitle={Proc. IEEE SLT},
  year={2022}
}
```
