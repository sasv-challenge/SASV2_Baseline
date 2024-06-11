# Adapted version -- Baseline of the ASVspoof 5 challenge Track 2, SASV.
- Adapted by Jee-weon Jung, Carnegie Mellon University.

This version is an adaptation to be used as a baseline in the [ASVspoof 5 challenge](https://www.asvspoof.org), track 2, Spoofing-robust Automatic Speaker Verification (SASV).
Following the first SASV challenge in Interspeech 2022, the second challenge has joined forces with the ASVspoof community and has became a track of the ASVspoof 5.
Using the very first large-scale data for speech anti-spoofing and deepfake, we once again aim to facilitate the development of single End-to-End (E2E) models that can reject spoofed inputs on top of the conventional non-target (different speaker) inputs.

In this baseline, we closly follow the approach proposed in [Towards single integrated spoofing-aware speaker verification embeddings](https://www.isca-archive.org/interspeech_2023/mun23_interspeech.pdf), which was presented at Interspeech 2023.
This adapted version trains a Deep Neural Network (DNN) that has two output layers, one for speaker identification (multi-class classification) and the other for anti-spoofing (bonafide/spoof classification).
- Pre-trained checkpoints and diverse other proposed training schemes are available in the main branch of this repository for those who are further interested.

Please send an email to `jeeweonj@ieee.org` for questions related to this adapted version for ASVspoof 5 challenge SASV track.

## Prerequisites

### Activate a conda environment
- Not mandatory, but I recommend to initialize a conda envionment and match the development environment. This is because Deep Neural Network (DNN) models' results are not deterministic when trained multiple times. Here is the environment I used.
```bash
conda create --name {asvspoof5_sasv2}
conda install -y conda "python=${3.9.19}"
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia\n
```
- Afterward, install the packages via `pip install -r requirements.txt`.

### Setting the corpus
1. Download `release.zip` by registering to the ASVspoof 5 challenge phase 2.
2. Add a symlink of your folder to this directory using below command.
  - `ln -s /path/of/your/uncompressed/release.zip corpus`

If you did it correctly, when you run `cd corpus; tree tree -I "*flac" ./`
```bash
./
|-- ASVspoof5.dev.enroll.txt
|-- ASVspoof5.dev.metadata.txt
|-- ASVspoof5.dev.trial.txt
|-- ASVspoof5.train.metadata.txt
|-- flac_D (derived using `tar -xvf flac_D.tar`)
|-- flac_D.tar
|   |--  D_0000000001.flac
|   |--  D_0000000002.flac
|   `--  D_...........flac
|-- flac_T ((derived using `tar -xvf flac_T.tar`))
|   |--  T_0000000001.flac
|   |--  T_0000000002.flac
|   `--  T_...........flac
|-- flac_T.tar
|-- LICENSE.txt
`-- README.txt
```

### Data format
- For the train set, we will use `ASVspoof5.train.metadata.txt`. There are six columns, where we will use `speeaker` and `spoofing` columns that inform who is speaking and whether it is a bonafide or spoofed.
  - We will adopt a multitask learning where a DNN is trained to classify both (i) closed set speaker identification (multi-class classification) and (ii) anti-spoofing detection (bonafide/spoof binary classification)

```
speaker utterance gender codec attack spoofing 
T_4850 T_0000000000 F - A05 spoof
```

- For the development set, we will use `ASVspoof5.dev.enroll.txt` and `ASVspoof5.dev.trial.txt`.
  - `ASVspoof5.dev.enroll.txt` has the information on which utterances comprise the speaker model (i.e., target or claimed speaker).
    - Example: `D_4660 D_A0000000316,D_A0000000725,D_A0000000343`
      - `D4660` is the speaker model's name and `D_A0000000316,D_A0000000725,D_A0000000343` are the three utterances that comprise the speaker.
  - `ASVspoof5.dev.trial.txt` is the evaluation protocol that denotes which speaker model to compare with which utterance and the ground truth (target/nontarget/spoof)
    - Example: `D_0755 D_0000000022 spoof`
    - The model is required to accept only the `target` and reject both `nontarget` and `spoof`.

## Models
This adaptation employs [SKA-TDNN](https://ieeexplore.ieee.org/iel7/10022052/10022330/10023305.pdf). You can also select other models implemented in the main branch (or your own model) using the `--model` option:

## Training
Train using the command below.

```bash
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
  --max_frames 400 \
  --num_spk 400 \
  --num_utt 2 \
  --batch_size 40 \
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
  --num_class 401 \
  --save_path exp/sasv_baseline \
  --train_list corpus/ASVspoof5.train.metadata.txt \
  --eval_list corpus/ASVspoof5.dev.trial.txt \
  --train_path corpus/flac_T \
  --eval_path corpus/flac_D \
  --spk_meta_train spk_meta/spk_meta_trn.pk \
  --spk_meta_eval spk_meta/spk_meta_dev.pk \
  --musan_path /path/to/dataset/MUSAN/musan_split \
  --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \
  --model SKA_TDNN
```

## Evaluation
You can evaluate your checkpoint of a model using:
```bash
CUDA_VISIBLE_DEVICES=0 python trainSASVNet.py \
        --eval \
        --eval_frames 0 \
        --num_eval 1 \
        --eval_list corpus/ASVspoof5.dev.trial.txt \
        --eval_path corpus/flac_D \
        --model SKA_TDNN \
        --initial_model /path/to/your_model/pretrained_weight.model
```

### Metric
We use the [Agnostic Detection Cost Function (a-DCF)](https://arxiv.org/abs/2403.01355) as the main metric, the primary metric used in the challenge to rank the submissions.

## Citation
If you utilize this repository, please cite the following papers,

```bibtex
@inproceedings{mun2022frequency,
  title={Frequency and Multi-Scale Selective Kernel Attention for Speaker Verification},
  author={Mun, Sung Hwan and Jung, Jee-weon and Han, Min Hyun and Kim, Nam Soo},
  booktitle={Proc. IEEE SLT},
  year={2022}
}
@inproceedings{mun2023towards
  title={Towards single integrated spoofing-aware speaker verification embeddings},
  author={Mun, Sung Hwan and Shim, Hye-jin and Tak, Hemlata and Wang, Xin and Liu, Xuechen and Sahidullah, Md and Jeong, Myeonghun and Han, Min Hyun and Todisco, Massimiliano and Lee, Kong Aik and others},
  booktitle={Proc. Interspeech},
  year={2023}
}
@inproceedings{shim2024an
  title={a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification},
  author={Shim, Hye-jin and Jung, Jee-weon and Kinnunen, Tomi and Evans, Nicholas and Bonastre, Jean-Francois and Lapidot, Itshak},
  booktitle={Proc. Speaker Odyssey},
  year={2024}
}
@techreport{delgado2024asvspoof,
  title={ASVspoof 5 Evaluation Plan},
  author={Delgado, H{\'e}ctor and Evans, Nicholas and Jung, Jee-weon and Kinnunen, Tomi and Kukanov, Ivan and Lee, Kong Aik and Liu, Xuechen and Shim, Hye-jin and Sahidullah, Md and Tak, Hemlata and others},
  year={2024},
  url={https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf},
}
```