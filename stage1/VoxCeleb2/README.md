# Stage 1

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


## Data Preparation
The [VoxCeleb](https://mm.kaist.ac.kr/datasets/voxceleb/) datasets are used for these experiments.
The train list should contain the file path and speaker identity for instance,
```
id00012/21Uxsk56VDQ/00001.wav id00012
id00012/21Uxsk56VDQ/00002.wav id00012
...
id09272/u7VNkYraCw0/00026.wav id09272
id09272/u7VNkYraCw0/00027.wav id09272
```
The example files of train list for VoxCeleb2 and the test lists for VoxCeleb1-O, VoxCeleb1-E, VoxCeleb1-H can be download from [train_vox2.txt](https://drive.google.com/file/d/1Y6yjKDULxJ40mhLzeKUzkeAvqNlP0tzX/view?usp=sharing) and [veri_test2.txt](https://drive.google.com/file/d/1EUDR5oCPC-zOexhLBHbFQpdnw1IRWq-B/view?usp=sharing), [list_test_all2](https://drive.google.com/file/d/1BgnEugORlSPsi4ZpTjTayAGPqyWTm7S8/view?usp=sharing), [list_test_hard2](https://drive.google.com/file/d/1p-gbPbDK4dy_SvSRWZ3KP17iZdHqjHQ4/view?usp=sharing), respectively. You can also follow the instructions on the [voxceleb_trainer](https://github.com/clovaai/voxceleb_trainer) repository for the download and data preparation of training, augmentation, and evaluation.

For the data augmentation of noise addition, you can download the [MUSAN noise corpus](https://www.openslr.org/17/).
After downloading and extracting the files, you can split the audio files into short segments for faster random access as the following command:
```bash
python process_musan.py /path/to/dataset/MUSAN
```
where `/path/to/dataset/MUSAN` is your path to the MUSAN corpus.

For the data augmentation of convolution with simulated RIRs, you can download the [Room Impulse Response and Noise Database](https://www.openslr.org/28/).


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
Distributed Data Parallel (DDP) training example: SKA_TDNN with a vanilla cosine similarity (COS) evaluation every epoch,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainSpeakerNet.py \
        --max_frames 200 \
        --eval_frames 0 \
        --num_eval 1 \
        --num_spk 100 \
        --num_utt 2 \
        --augment Ture \
        --optimizer adamW \
        --scheduler cosine_annealing_warmup_restarts \
        --lr_t0 25 \
        --lr_tmul 1.0 \
        --lr_max 1e-3 \
        --lr_min 1e-8 \
        --lr_wstep 10 \
        --lr_gamma 0.5 \
        --margin 0.2 \
        --scale 30 \
        --num_class 5994 \
        --save_path ./save/ska_tdnn \
        --train_list ./list/train_vox2.txt \
        --test_list ./list/veri_test2.txt \
        --train_path /path/to/dataset/VoxCeleb2/dev/wav \
        --test_path /path/to/dataset/VoxCeleb1/test/wav \
        --musan_path /path/to/dataset/MUSAN/musan_split \
        --rir_path /path/to/dataset/RIRS_NOISES/simulated_rirs \
        --model SKA_TDNN \
        --port 8000 \
        --distributed
```

## Evaluation
Evaluation example using vanilla cosine similarity (COS) on the VoxCeleb1-O,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainSpeakerNet.py \
        --eval \
        --eval_frames 0 \
        --num_eval 1 \
        --initial_model ./save/model/your_model.model \
        --test_list ./list/veri_test2.txt \
        --test_path /path/to/dataset/VoxCeleb1/test/wav \
        --model SKA_TDNN \
        --port 8001 \
        --distributed
```
Evaluation example using Test Time Augmentation (TTA) on the VoxCeleb1-E,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainSpeakerNet.py \
        --eval \
        --tta \
        --eval_frames 400 \
        --num_eval 10 \
        --initial_model ./save/model/your_model.model \
        --test_list ./list/list_test_all2 \
        --test_path /path/to/dataset/VoxCeleb1/all/wav \
        --model SKA_TDNN \
        --port 8002 \
        --distributed
```
Evaluation example using Score Normalisation (SN) on the VoxCeleb1-H,
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python trainSpeakerNet.py \
        --eval \
        --score_norm \
        --type_coh utt \
        --top_coh_size 20000 \
        --eval_frames 0 \
        --num_eval 1 \
        --initial_model ./save/model/your_model.model \
        --train_list ./list/train_vox2.txt \
        --test_list ./list/list_test_hard2 \
        --train_path /path/to/dataset/VoxCeleb2/dev/wav \
        --test_path /path/to/dataset/VoxCeleb1/all/wav \
        --model SKA_TDNN \
        --port 8003 \
        --distributed
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