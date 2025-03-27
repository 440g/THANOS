```
(HAT) lina4544@DESKTOP-2HN27IT:~/HAT$ python evo_search.py --configs=configs/wmt14.en-de/supertransformer/space0.yml --evo-configs=configs/wmt14.en-de/evo_search/wmt14ende_npu.yml
Namespace(configs='configs/wmt14.en-de/supertransformer/space0.yml', pdb=False, no_progress_bar=False, log_interval=1000, log_format=None, tensorboard_logdir='checkpoints/wmt14.en-de/supertransformer/space0/tensorboard', tbmf_wrapper=False, seed=1, cpu=False, fp16=True, memory_efficient_fp16=False, fp16_init_scale=128, fp16_scale_window=None, fp16_scale_tolerance=0.0, min_loss_scale=0.0001, threshold_loss_scale=None, user_dir=None, criterion='label_smoothed_cross_entropy', optimizer='adam', lr_scheduler='cosine', task='translation', num_workers=10, skip_invalid_size_inputs_valid_test=False, max_tokens=4096, max_sentences=None, required_batch_size_multiple=8, dataset_impl=None, train_subset='train', valid_subset='valid', validate_interval=10, disable_validation=False, max_tokens_valid=4096, max_sentences_valid=None, curriculum=0, distributed_world_size=1, distributed_rank=0, distributed_backend='nccl', distributed_init_method=None, distributed_port=-1, device_id=0, distributed_no_spawn=False, ddp_backend='no_c10d', bucket_cap_mb=25, fix_batches_to_gpus=False, find_unused_parameters=False, arch='transformersuper_wmt_en_de', max_epoch=0, max_update=40000, clip_norm=0.0, sentence_avg=False, update_freq=[16], lr=[1e-07], min_lr=-1, use_bmuf=False, save_dir='checkpoints/wmt14.en-de/supertransformer/space0', restore_file='./downloaded_models/HAT_wmt14ende_super_space0.pt', reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, optimizer_overrides='{}', save_interval=10, save_interval_updates=0, keep_interval_updates=-1, keep_last_epochs=20, no_save=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_save_optimizer_state=False, best_checkpoint_metric='loss', maximize_best_checkpoint_metric=False, evo_configs='configs/wmt14.en-de/evo_search/wmt14ende_npu.yml', evo_iter=30, population_size=125, parent_size=25, mutation_size=50, crossover_size=50, mutation_prob=0.3, feature_norm=[640.0, 6.0, 2048.0, 6.0, 640.0, 6.0, 2048.0, 6.0, 6.0, 2.0], lat_norm=300.0, ckpt_path='./latency_dataset/predictors/wmt14ende_npu.pt', latency_constraint=200.0, valid_cnt_max=1000000000.0, write_config_path='configs/wmt14.en-de/subtransformer/wmt14ende_npu@200ms.yml', path=None, remove_bpe=None, quiet=False, model_overrides='{}', results_path=None, beam=5, nbest=1, max_len_a=0, max_len_b=200, min_len=1, match_source_len=False, no_early_stop=False, unnormalized=False, no_beamable_mm=False, lenpen=1, unkpen=0, replace_unk=None, sacrebleu=False, score_reference=False, prefix_size=0, no_repeat_ngram_size=0, sampling=False, sampling_topk=-1, sampling_topp=-1.0, temperature=1.0, diverse_beam_groups=-1, diverse_beam_strength=0.5, print_alignment=False, profile_latency=False, no_token_positional_embeddings=False, get_attn=False, encoder_embed_choice=[640, 512], decoder_embed_choice=[640, 512], encoder_layer_num_choice=[6], decoder_layer_num_choice=[6, 5, 4, 3, 2, 1], encoder_ffn_embed_dim_choice=[3072, 2048, 1024], decoder_ffn_embed_dim_choice=[3072, 2048, 1024], encoder_self_attention_heads_choice=[8, 4], decoder_self_attention_heads_choice=[8, 4], decoder_ende_attention_heads_choice=[8, 4], qkv_dim=512, decoder_arbitrary_ende_attn_choice=[-1, 1, 2], vocab_original_scaling=False, encoder_embed_dim_subtransformer=None, decoder_embed_dim_subtransformer=None, encoder_ffn_embed_dim_all_subtransformer=None, decoder_ffn_embed_dim_all_subtransformer=None, encoder_self_attention_heads_all_subtransformer=None, decoder_self_attention_heads_all_subtransformer=None, decoder_ende_attention_heads_all_subtransformer=None, decoder_arbitrary_ende_attn_all_subtransformer=None, label_smoothing=0.1, adam_betas='(0.9, 0.98)', adam_eps=1e-08, weight_decay=0.0, warmup_updates=10000, warmup_init_lr=1e-07, max_lr=0.001, t_mult=1, lr_period_updates=-1, lr_shrink=1.0, data='data/binary/wmt16_en_de', source_lang=None, target_lang=None, lazy_load=False, raw_text=False, left_pad_source='True', left_pad_target='False', max_source_positions=1024, max_target_positions=1024, upsample_primary=1, share_all_embeddings=True, dropout=0.3, attention_dropout=0.1, encoder_embed_dim=640, decoder_embed_dim=640, encoder_ffn_embed_dim=3072, decoder_ffn_embed_dim=3072, encoder_layers=6, decoder_layers=6, encoder_attention_heads=8, decoder_attention_heads=8, encoder_embed_path=None, encoder_normalize_before=False, encoder_learned_pos=False, decoder_embed_path=None, decoder_normalize_before=False, decoder_learned_pos=False, activation_dropout=0.0, activation_fn='relu', adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, share_decoder_input_output_embed=False, adaptive_input=False, decoder_output_dim=640, decoder_input_dim=640)
| [en] dictionary: 32768 types
| [de] dictionary: 32768 types
| loaded 3000 examples from: data/binary/wmt16_en_de/valid.en-de.en
| loaded 3000 examples from: data/binary/wmt16_en_de/valid.en-de.de
| data/binary/wmt16_en_de valid en-de 3000 examples
| Fallback to xavier initializer
TransformerSuperModel(
  (encoder): TransformerEncoder(
    (embed_tokens): EmbeddingSuper(32768, 640, padding_idx=1)
    (embed_positions): SinusoidalPositionalEmbedding()
    (layers): ModuleList(
      (0-5): 6 x TransformerEncoderLayer(
        (self_attn): MultiheadAttentionSuper    num_heads:8      qkv_dim:512
          (out_proj): LinearSuper(in_features=512, out_features=640, bias=True)
        )
        (self_attn_layer_norm): LayerNormSuper((640,), eps=1e-05, elementwise_affine=True)
        (fc1): LinearSuper(in_features=640, out_features=3072, bias=True)
        (fc2): LinearSuper(in_features=3072, out_features=640, bias=True)
        (final_layer_norm): LayerNormSuper((640,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (decoder): TransformerDecoder(
    (embed_tokens): EmbeddingSuper(32768, 640, padding_idx=1)
    (embed_positions): SinusoidalPositionalEmbedding()
    (layers): ModuleList(
      (0-5): 6 x TransformerDecoderLayer(
        (self_attn): MultiheadAttentionSuper    num_heads:8      qkv_dim:512
          (out_proj): LinearSuper(in_features=512, out_features=640, bias=True)
        )
        (self_attn_layer_norm): LayerNormSuper((640,), eps=1e-05, elementwise_affine=True)
        (encoder_attn): MultiheadAttentionSuper num_heads:8      qkv_dim:512
          (out_proj): LinearSuper(in_features=512, out_features=640, bias=True)
        )
        (encoder_attn_layer_norm): LayerNormSuper((640,), eps=1e-05, elementwise_affine=True)
        (fc1): LinearSuper(in_features=640, out_features=3072, bias=True)
        (fc2): LinearSuper(in_features=3072, out_features=640, bias=True)
        (final_layer_norm): LayerNormSuper((640,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
)
/home/lina4544/HAT/fairseq/checkpoint_utils.py:137: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load(
| loaded checkpoint ./downloaded_models/HAT_wmt14ende_super_space0.pt (epoch 136 @ 0 updates)
| loading train data for epoch 136
| loaded 3000 examples from: data/binary/wmt16_en_de/valid.en-de.en
| loaded 3000 examples from: data/binary/wmt16_en_de/valid.en-de.de
| data/binary/wmt16_en_de valid en-de 3000 examples
/home/lina4544/HAT/latency_predictor.py:111: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.model.load_state_dict(torch.load(self.ckpt_path))
| Start Iteration 0:
/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/utils/data/dataloader.py:557: UserWarning: This DataLoader will create 10 worker processes in total. Our suggested max number of worker in current system is 8, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
| epoch 136 | valid on 'valid' subset:   7%|██▊                                       | 2/30 [08:15<1:| epoch 136 | valid on 'valid' subset:  10%|████▏                                     | 3/30 [12:37<1:| epoch 136 | valid on 'valid' subset:  13%|█████▌                                    | 4/30 [16:44<1:| epoch 136 | valid on 'valid' subset:  17%|███████                                   | 5/30 [20:40<1:| epoch 136 | valid on 'valid' subset:  20%|████████▍                                 | 6/30 [24:34<1:| epoch 136 | valid on 'valid' subset:  23%|█████████▊                                | 7/30 [28:47<1:| epoch 136 | valid on 'valid' subset:  27%|███████████▏                              | 8/30 [32:52<1:| epoch 136 | valid on 'valid' subset:  30%|████████████▌                             | 9/30 [37:07<1:| epoch 136 | valid on 'valid' subset:  33%|█████████████▋                           | 10/30 [41:18<1:| epoch 136 | valid on 'valid' subset:  37%|███████████████                          | 11/30 [45:20<1:| epoch 136 | valid on 'valid' subset:  40%|████████████████▍                        | 12/30 [49:20<1:| epoch 136 | valid on 'valid' subset:  43%|█████████████████▊                       | 13/30 [53:33<1:| epoch 136 | valid on 'valid' subset:  47%|███████████████████▏                     | 14/30 [57:39<1:| epoch 136 | valid on 'valid' subset:  50%|███████████████████▌                   | 15/30 [1:01:38<1:| epoch 136 | valid on 'valid' subset:  53%|█████████████████████▊                   | 16/30 [1:05:47<| epoch 136 | valid on 'valid' subset:  57%|███████████████████████▏                 | 17/30 [1:09:39<| epoch 136 | valid on 'valid' subset:  60%|████████████████████████▌                | 18/30 [1:13:43<| epoch 136 | valid on 'valid' subset:  63%|█████████████████████████▉               | 19/30 [1:17:57<| epoch 136 | valid on 'valid' subset:  67%|███████████████████████████▎             | 20/30 [1:22:10<| epoch 136 | valid on 'valid' subset:  70%|████████████████████████████▋            | 21/30 [1:26:11<| epoch 136 | valid on 'valid' subset:  73%|██████████████████████████████           | 22/30 [1:30:07<| epoch 136 | valid on 'valid' subset:  77%|███████████████████████████████▍         | 23/30 [1:34:08<| epoch 136 | valid on 'valid' subset:  80%|████████████████████████████████▊        | 24/30 [1:38:26<| epoch 136 | valid on 'valid' subset:  83%|██████████████████████████████████▏      | 25/30 [1:42:43<| epoch 136 | valid on 'valid' subset:  87%|███████████████████████████████████▌     | 26/30 [1:47:01<| epoch 136 | valid on 'valid' subset:  90%|████████████████████████████████████▉    | 27/30 [1:51:05<| epoch 136 | valid on 'valid' subset:  93%|██████████████████████████████████████▎  | 28/30 [1:55:30<| epoch 136 | valid on 'valid' subset:  97%|███████████████████████████████████████▋ | 29/30 [1:58:59<| epoch 136 | valid on 'valid' subset: 100%|█████████████████████████████████████████| 30/30 [2:02:43<
| Iteration 0, Lowest loss: 7.82304145626689
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 512, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 3072, 2048, 2048, 1024, 3072], 'encoder_self_attention_heads': [8, 4, 4, 4, 4, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 5, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 1024, 1024, 1024], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 4, 4, 4], 'decoder_arbitrary_ende_attn': [2, 1, -1, 2, -1, 1]}}
| Predicted latency for lowest loss model: 2.1135762333869934
| Start Iteration 1:
| epoch 136 | valid on 'valid' subset:  97%|██████████████████████▏| 29/30 [2:05:20<04:13, 253.61s/it]Traceback (most recent call last):
  File "/home/lina4544/HAT/evo_search.py", line 110, in <module>
    cli_main()
  File "/home/lina4544/HAT/evo_search.py", line 106, in cli_main
    main(args)
  File "/home/lina4544/HAT/evo_search.py", line 51, in main
    best_config = evolver.run_evo_search()
  File "/home/lina4544/HAT/fairseq/evolution.py", line 217, in run_evo_search
    popu_scores = self.get_scores(popu)
  File "/home/lina4544/HAT/fairseq/evolution.py", line 281, in get_scores
    scores = validate_all(self.args, self.trainer, self.task, self.epoch_iter, configs)
  File "/home/lina4544/HAT/fairseq/evolution.py", line 401, in validate_all
    trainer.valid_step(sample)
  File "/home/lina4544/HAT/fairseq/trainer.py", line 438, in valid_step
    _loss, sample_size, logging_output = self.task.valid_step(
  File "/home/lina4544/HAT/fairseq/tasks/fairseq_task.py", line 241, in valid_step
    loss, sample_size, logging_output = criterion(model, sample)
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/lina4544/HAT/fairseq/criterions/label_smoothed_cross_entropy.py", line 56, in forward
    net_output = model(**sample['net_input'])
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/lina4544/HAT/fairseq/models/fairseq_model.py", line 223, in forward
    decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/lina4544/HAT/fairseq/models/transformer_super.py", line 622, in forward
    x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
  File "/home/lina4544/HAT/fairseq/models/transformer_super.py", line 693, in extract_features
    x, attn = layer(
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/lina4544/HAT/fairseq/models/transformer_super.py", line 1123, in forward
    x = self.fc2(x)
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/lina4544/HAT/fairseq/modules/linear_super.py", line 58, in forward
    return F.linear(x, self.samples['weight'], self.samples['bias'])
KeyboardInterrupt
```