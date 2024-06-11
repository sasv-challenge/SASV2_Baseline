Namespace(max_frames=500, eval_frames=0, num_eval=1, num_spk=40, num_utt=2, batch_size=160, max_seg_per_spk=10000, num_thread=10, augment=False, seed=10, test_interval=1, max_epoch=100, trainfunc='sasv_e2e_v1', optimizer='adamW', scheduler='cosine_annealing_warmup_restarts', weight_decay=1e-07, lr=0.0001, lr_t0=8, lr_tmul=1.0, lr_max=0.0001, lr_min=0.0, lr_wstep=0, lr_gamma=0.8, margin=0.2, scale=30.0, num_class=41, initial_model='', save_path='exp/sasv_baseline', train_list='corpus/ASVspoof5.train.metadata.txt', eval_list='corpus/ASVspoof5.dev.trial.txt', train_path='corpus/flac_T', eval_path='corpus/flac_D', spk_meta_train='spk_meta/spk_meta_trn.pk', spk_meta_eval='spk_meta/spk_meta_dev.pk', musan_path='/home/jeeweonj/corpora/voxcelebs/musan', rir_path='/home/jeeweonj/corpora/voxcelebs/RIRS_NOISES/simulated_rirs', num_mels=80, log_input=True, model='SKA_TDNN', pooling_type='ASP', num_out=192, eca_c=1024, eca_s=8, eval=False, scoring=False, model_save_path='exp/sasv_baseline/model', result_save_path='exp/sasv_baseline/result')