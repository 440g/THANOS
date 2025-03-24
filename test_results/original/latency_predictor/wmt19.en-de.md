```
(HAT) lina4544@DESKTOP-2HN27IT:~/HAT$ python evo_search.py --configs=configs/wmt19.en-de/supertransformer/space0.yml --evo-configs=configs/wmt19.en-de/evo_search/wmt19ende_npu.yml
Namespace(configs='configs/wmt19.en-de/supertransformer/space0.yml', pdb=False, no_progress_bar=False, log_interval=1000, log_format=None, tensorboard_logdir='checkpoints/wmt19.en-de/supertransformer/space0/tensorboard', tbmf_wrapper=False, seed=1, cpu=False, fp16=True, memory_efficient_fp16=False, fp16_init_scale=128, fp16_scale_window=None, fp16_scale_tolerance=0.0, min_loss_scale=0.0001, threshold_loss_scale=None, user_dir=None, criterion='label_smoothed_cross_entropy', optimizer='adam', lr_scheduler='cosine', task='translation', num_workers=10, skip_invalid_size_inputs_valid_test=False, max_tokens=4096, max_sentences=None, required_batch_size_multiple=8, dataset_impl=None, train_subset='train', valid_subset='valid', validate_interval=1, disable_validation=False, max_tokens_valid=4096, max_sentences_valid=None, curriculum=0, distributed_world_size=1, distributed_rank=0, distributed_backend='nccl', distributed_init_method=None, distributed_port=-1, device_id=0, distributed_no_spawn=False, ddp_backend='no_c10d', bucket_cap_mb=25, fix_batches_to_gpus=False, find_unused_parameters=False, arch='transformersuper_wmt_en_de', max_epoch=0, max_update=40000, clip_norm=0.0, sentence_avg=False, update_freq=[16], lr=[1e-07], min_lr=-1, use_bmuf=False, save_dir='checkpoints/wmt19.en-de/supertransformer/space0', restore_file='./downloaded_models/HAT_wmt19ende_super_space0.pt', reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, optimizer_overrides='{}', save_interval=1, save_interval_updates=0, keep_interval_updates=-1, keep_last_epochs=20, no_save=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_save_optimizer_state=False, best_checkpoint_metric='loss', maximize_best_checkpoint_metric=False, evo_configs='configs/wmt19.en-de/evo_search/wmt19ende_npu.yml', evo_iter=30, population_size=125, parent_size=25, mutation_size=50, crossover_size=50, mutation_prob=0.3, feature_norm=[640.0, 6.0, 2048.0, 6.0, 640.0, 6.0, 2048.0, 6.0, 6.0, 2.0], lat_norm=200.0, ckpt_path='./latency_dataset/predictors/wmt19ende_npu.pt', latency_constraint=200.0, valid_cnt_max=1000000000.0, write_config_path='configs/wmt19.en-de/subtransformer/wmt19ende_npu@200ms.yml', path=None, remove_bpe=None, quiet=False, model_overrides='{}', results_path=None, beam=5, nbest=1, max_len_a=0, max_len_b=200, min_len=1, match_source_len=False, no_early_stop=False, unnormalized=False, no_beamable_mm=False, lenpen=1, unkpen=0, replace_unk=None, sacrebleu=False, score_reference=False, prefix_size=0, no_repeat_ngram_size=0, sampling=False, sampling_topk=-1, sampling_topp=-1.0, temperature=1.0, diverse_beam_groups=-1, diverse_beam_strength=0.5, print_alignment=False, profile_latency=False, no_token_positional_embeddings=False, get_attn=False, encoder_embed_choice=[640, 512], decoder_embed_choice=[640, 512], encoder_layer_num_choice=[6], decoder_layer_num_choice=[6, 5, 4, 3, 2, 1], encoder_ffn_embed_dim_choice=[3072, 2048, 1024], decoder_ffn_embed_dim_choice=[3072, 2048, 1024], encoder_self_attention_heads_choice=[8, 4], decoder_self_attention_heads_choice=[8, 4], decoder_ende_attention_heads_choice=[8, 4], qkv_dim=512, decoder_arbitrary_ende_attn_choice=[-1, 1, 2], vocab_original_scaling=False, encoder_embed_dim_subtransformer=None, decoder_embed_dim_subtransformer=None, encoder_ffn_embed_dim_all_subtransformer=None, decoder_ffn_embed_dim_all_subtransformer=None, encoder_self_attention_heads_all_subtransformer=None, decoder_self_attention_heads_all_subtransformer=None, decoder_ende_attention_heads_all_subtransformer=None, decoder_arbitrary_ende_attn_all_subtransformer=None, label_smoothing=0.1, adam_betas='(0.9, 0.98)', adam_eps=1e-08, weight_decay=0.0, warmup_updates=10000, warmup_init_lr=1e-07, max_lr=0.001, t_mult=1, lr_period_updates=-1, lr_shrink=1.0, data='data/binary/wmt19_en_de', source_lang=None, target_lang=None, lazy_load=False, raw_text=False, left_pad_source='True', left_pad_target='False', max_source_positions=1024, max_target_positions=1024, upsample_primary=1, share_all_embeddings=True, dropout=0.3, attention_dropout=0.1, encoder_embed_dim=640, decoder_embed_dim=640, encoder_ffn_embed_dim=3072, decoder_ffn_embed_dim=3072, encoder_layers=6, decoder_layers=6, encoder_attention_heads=8, decoder_attention_heads=8, encoder_embed_path=None, encoder_normalize_before=False, encoder_learned_pos=False, decoder_embed_path=None, decoder_normalize_before=False, decoder_learned_pos=False, activation_dropout=0.0, activation_fn='relu', adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, share_decoder_input_output_embed=False, adaptive_input=False, decoder_output_dim=640, decoder_input_dim=640)
| [en] dictionary: 49600 types
| [de] dictionary: 49600 types
| loaded 2942 examples from: data/binary/wmt19_en_de/valid.en-de.en
| loaded 2942 examples from: data/binary/wmt19_en_de/valid.en-de.de
| data/binary/wmt19_en_de valid en-de 2942 examples
| Fallback to xavier initializer
TransformerSuperModel(
  (encoder): TransformerEncoder(
    (embed_tokens): EmbeddingSuper(49600, 640, padding_idx=1)
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
    (embed_tokens): EmbeddingSuper(49600, 640, padding_idx=1)
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
| loaded checkpoint ./downloaded_models/HAT_wmt19ende_super_space0.pt (epoch 13 @ 0 updates)
| loading train data for epoch 13
| loaded 2942 examples from: data/binary/wmt19_en_de/valid.en-de.en
| loaded 2942 examples from: data/binary/wmt19_en_de/valid.en-de.de
| data/binary/wmt19_en_de valid en-de 2942 examples
/home/lina4544/HAT/latency_predictor.py:111: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.model.load_state_dict(torch.load(self.ckpt_path))
| Start Iteration 0:
/home/lina4544/miniconda3/envs/HAT/lib/python3.9/site-packages/torch/utils/data/dataloader.py:557: UserWarning: This DataLoader will create 10 worker processes in total. Our suggested max number of worker in current system is 8, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(_create_warning_msg(
| epoch 013 | valid on 'valid' subset:   4%|█▌                                        | 1/28 [06:23<2:| epoch 013 | valid on 'valid' subset:   7%|███                                       | 2/28 [12:12<2:| epoch 013 | valid on 'valid' subset:  11%|████▌                                     | 3/28 [18:04<2:| epoch 013 | valid on 'valid' subset:  14%|██████                                    | 4/28 [23:51<2:| epoch 013 | valid on 'valid' subset:  18%|███████▌                                  | 5/28 [29:35<2:| epoch 013 | valid on 'valid' subset:  21%|█████████                                 | 6/28 [35:21<2:| epoch 013 | valid on 'valid' subset:  25%|██████████▌                               | 7/28 [41:03<2:| epoch 013 | valid on 'valid' subset:  29%|████████████                              | 8/28 [46:45<1:| epoch 013 | valid on 'valid' subset:  32%|█████████████▌                            | 9/28 [52:35<1:| epoch 013 | valid on 'valid' subset:  36%|██████████████▋                          | 10/28 [58:09<1:| epoch 013 | valid on 'valid' subset:  39%|███████████████▎                       | 11/28 [1:04:03<1:| epoch 013 | valid on 'valid' subset:  43%|████████████████▋                      | 12/28 [1:09:47<1:| epoch 013 | valid on 'valid' subset:  46%|██████████████████                     | 13/28 [1:15:40<1:| epoch 013 | valid on 'valid' subset:  50%|███████████████████▌                   | 14/28 [1:21:03<1:| epoch 013 | valid on 'valid' subset:  54%|████████████████████▉                  | 15/28 [1:26:43<1:| epoch 013 | valid on 'valid' subset:  57%|██████████████████████▎                | 16/28 [1:32:12<1:| epoch 013 | valid on 'valid' subset:  61%|███████████████████████▋               | 17/28 [1:37:36<1:| epoch 013 | valid on 'valid' subset:  64%|██████████████████████████▎              | 18/28 [1:43:10<| epoch 013 | valid on 'valid' subset:  68%|███████████████████████████▊             | 19/28 [1:48:43<| epoch 013 | valid on 'valid' subset:  71%|█████████████████████████████▎           | 20/28 [1:54:23<| epoch 013 | valid on 'valid' subset:  75%|██████████████████████████████▊          | 21/28 [2:00:13<| epoch 013 | valid on 'valid' subset:  79%|████████████████████████████████▏        | 22/28 [2:06:07<| epoch 013 | valid on 'valid' subset:  82%|█████████████████████████████████▋       | 23/28 [2:12:09<| epoch 013 | valid on 'valid' subset:  86%|███████████████████████████████████▏     | 24/28 [2:17:35<| epoch 013 | valid on 'valid' subset:  89%|████████████████████████████████████▌    | 25/28 [2:23:27<| epoch 013 | valid on 'valid' subset:  93%|██████████████████████████████████████   | 26/28 [2:28:28<| epoch 013 | valid on 'valid' subset:  96%|███████████████████████████████████████▌ | 27/28 [2:34:45<| epoch 013 | valid on 'valid' subset: 100%|█████████████████████████████████████████| 28/28 [2:35:43<
| Iteration 0, Lowest loss: 7.322309718777086
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 3072, 2048, 1024, 2048, 3072], 'encoder_self_attention_heads': [8, 8, 8, 4, 4, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 2048, 2048, 1024, 2048, 1024], 'decoder_self_attention_heads': [8, 8, 8, 4, 8, 8], 'decoder_ende_attention_heads': [4, 4, 8, 4, 4, 4], 'decoder_arbitrary_ende_attn': [1, 2, -1, 2, 2, -1]}}
| Predicted latency for lowest loss model: 41.66412353515625
| Start Iteration 1:
| epoch 013 | valid on 'valid' subset:  36%|████████▏              | 10/28 [44:59<1:20:47, 269.29s/it]Traceback (most recent call last):
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
  File "/home/lina4544/HAT/fairseq/models/transformer_super.py", line 623, in forward
    x = self.output_layer(x)
  File "/home/lina4544/HAT/fairseq/models/transformer_super.py", line 721, in output_layer
    return F.linear(features, self.embed_tokens.sampled_weight('decoder'))
KeyboardInterrupt
```