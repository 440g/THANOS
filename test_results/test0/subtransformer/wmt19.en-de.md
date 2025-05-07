```
(SH) suhyeon@DESKTOP-2HN27IT:~/myHAT$ python evo_search.py --configs=configs/wmt19.en-de/supertransformer/space0.yml --evo-configs=configs/wmt19.en-de/evo_search/wmt19ende_npu.yml --num-workers=8 --cpu
Namespace(configs='configs/wmt19.en-de/supertransformer/space0.yml', pdb=False, no_progress_bar=False, log_interval=1000, log_format=None, tensorboard_logdir='checkpoints/wmt19.en-de/supertransformer/space0/tensorboard', tbmf_wrapper=False, seed=1, cpu=True, fp16=False, memory_efficient_fp16=False, fp16_init_scale=128, fp16_scale_window=None, fp16_scale_tolerance=0.0, min_loss_scale=0.0001, threshold_loss_scale=None, user_dir=None, criterion='label_smoothed_cross_entropy', optimizer='adam', lr_scheduler='cosine', task='translation', num_workers=8, skip_invalid_size_inputs_valid_test=False, max_tokens=4096, max_sentences=None, required_batch_size_multiple=8, dataset_impl=None, train_subset='train', valid_subset='valid', validate_interval=1, disable_validation=False, max_tokens_valid=4096, max_sentences_valid=None, curriculum=0, distributed_world_size=1, distributed_rank=0, distributed_backend='nccl', distributed_init_method=None, distributed_port=-1, device_id=0, distributed_no_spawn=False, ddp_backend='no_c10d', bucket_cap_mb=25, fix_batches_to_gpus=False, find_unused_parameters=False, arch='transformersuper_wmt_en_de', max_epoch=0, max_update=40000, clip_norm=0.0, sentence_avg=False, update_freq=[16], lr=[1e-07], min_lr=-1, use_bmuf=False, save_dir='checkpoints/wmt19.en-de/supertransformer/space0', restore_file='./downloaded_models/HAT_wmt19ende_super_space0.pt', reset_dataloader=False, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, optimizer_overrides='{}', save_interval=1, save_interval_updates=0, keep_interval_updates=-1, keep_last_epochs=20, no_save=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_save_optimizer_state=False, best_checkpoint_metric='loss', maximize_best_checkpoint_metric=False, evo_configs='configs/wmt19.en-de/evo_search/wmt19ende_npu.yml', evo_iter=30, population_size=125, parent_size=25, mutation_size=50, crossover_size=50, mutation_prob=0.3, feature_norm=[640.0, 6.0, 2048.0, 6.0, 640.0, 6.0, 2048.0, 6.0, 6.0, 2.0], lat_norm=200.0, ckpt_path='./latency_dataset/predictors/wmt19ende_npu.pt', latency_constraint=200.0, valid_cnt_max=1000000000.0, write_config_path='configs/wmt19.en-de/subtransformer/wmt19ende_npu@200ms.yml', path=None, remove_bpe=None, quiet=False, model_overrides='{}', results_path=None, beam=5, nbest=1, max_len_a=0, max_len_b=200, min_len=1, match_source_len=False, no_early_stop=False, unnormalized=False, no_beamable_mm=False, lenpen=1, unkpen=0, replace_unk=None, sacrebleu=False, score_reference=False, prefix_size=0, no_repeat_ngram_size=0, sampling=False, sampling_topk=-1, sampling_topp=-1.0, temperature=1.0, diverse_beam_groups=-1, diverse_beam_strength=0.5, print_alignment=False, profile_latency=False, no_token_positional_embeddings=False, get_attn=False, encoder_embed_choice=[640, 512], decoder_embed_choice=[640, 512], encoder_layer_num_choice=[6], decoder_layer_num_choice=[6, 5, 4, 3, 2, 1], encoder_ffn_embed_dim_choice=[3072, 2048, 1024], decoder_ffn_embed_dim_choice=[3072, 2048, 1024], encoder_self_attention_heads_choice=[8, 4], decoder_self_attention_heads_choice=[8, 4], decoder_ende_attention_heads_choice=[8, 4], qkv_dim=512, decoder_arbitrary_ende_attn_choice=[-1, 1, 2], vocab_original_scaling=False, encoder_embed_dim_subtransformer=None, decoder_embed_dim_subtransformer=None, encoder_ffn_embed_dim_all_subtransformer=None, decoder_ffn_embed_dim_all_subtransformer=None, encoder_self_attention_heads_all_subtransformer=None, decoder_self_attention_heads_all_subtransformer=None, decoder_ende_attention_heads_all_subtransformer=None, decoder_arbitrary_ende_attn_all_subtransformer=None, label_smoothing=0.1, adam_betas='(0.9, 0.98)', adam_eps=1e-08, weight_decay=0.0, warmup_updates=10000, warmup_init_lr=1e-07, max_lr=0.001, t_mult=1, lr_period_updates=-1, lr_shrink=1.0, data='data/binary/wmt19_en_de', source_lang=None, target_lang=None, lazy_load=False, raw_text=False, left_pad_source='True', left_pad_target='False', max_source_positions=1024, max_target_positions=1024, upsample_primary=1, share_all_embeddings=True, dropout=0.3, attention_dropout=0.1, encoder_embed_dim=640, decoder_embed_dim=640, encoder_ffn_embed_dim=3072, decoder_ffn_embed_dim=3072, encoder_layers=6, decoder_layers=6, encoder_attention_heads=8, decoder_attention_heads=8, encoder_embed_path=None, encoder_normalize_before=False, encoder_learned_pos=False, decoder_embed_path=None, decoder_normalize_before=False, decoder_learned_pos=False, activation_dropout=0.0, activation_fn='relu', adaptive_softmax_cutoff=None, adaptive_softmax_dropout=0, share_decoder_input_output_embed=False, adaptive_input=False, decoder_output_dim=640, decoder_input_dim=640)
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
/home/suhyeon/myHAT/fairseq/checkpoint_utils.py:137: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  state = torch.load(
| loaded checkpoint ./downloaded_models/HAT_wmt19ende_super_space0.pt (epoch 13 @ 0 updates)
| loading train data for epoch 13
| loaded 2942 examples from: data/binary/wmt19_en_de/valid.en-de.en
| loaded 2942 examples from: data/binary/wmt19_en_de/valid.en-de.de
| data/binary/wmt19_en_de valid en-de 2942 examples
/home/suhyeon/myHAT/latency_predictor.py:111: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.model.load_state_dict(torch.load(self.ckpt_path))
| Start Iteration 0:
| Iteration 0, Lowest loss: 7.28997254452245
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 2048], 'encoder_self_attention_heads': [8, 4, 4, 8, 4, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 5, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 8, 8, 4, 4], 'decoder_ende_attention_heads': [4, 8, 8, 4, 4, 8], 'decoder_arbitrary_ende_attn': [2, 2, 2, 2, 2, -1]}}
| Predicted latency for lowest loss model: 19.423839449882507
| Start Iteration 1:
| Iteration 1, Lowest loss: 7.28997254452245
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 2048], 'encoder_self_attention_heads': [8, 4, 4, 8, 4, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 5, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 8, 8, 4, 4], 'decoder_ende_attention_heads': [4, 8, 8, 4, 4, 8], 'decoder_arbitrary_ende_attn': [2, 2, 2, 2, 2, -1]}}
| Predicted latency for lowest loss model: 19.423839449882507
| Start Iteration 2:
| Iteration 2, Lowest loss: 7.2355617557879865
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 4, 8, 8, 8, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 1024, 1024, 1024], 'decoder_self_attention_heads': [8, 8, 4, 4, 4, 8], 'decoder_ende_attention_heads': [4, 4, 4, 4, 8, 8], 'decoder_arbitrary_ende_attn': [2, 2, 1, 2, 1, 1]}}
| Predicted latency for lowest loss model: 18.956050276756287
| Start Iteration 3:
| Iteration 3, Lowest loss: 7.224946976714659
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 1024, 2048, 3072, 1024], 'encoder_self_attention_heads': [8, 8, 8, 8, 8, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 5, 'decoder_ffn_embed_dim': [1024, 3072, 1024, 3072, 2048, 2048], 'decoder_self_attention_heads': [4, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 4, 8, 8], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 1]}}
| Predicted latency for lowest loss model: 19.208134710788727
| Start Iteration 4:
| Iteration 4, Lowest loss: 7.206420330657853
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 1024, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 2048, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 8], 'decoder_ende_attention_heads': [4, 4, 4, 4, 8, 8], 'decoder_arbitrary_ende_attn': [2, 2, 1, 2, 2, 1]}}
| Predicted latency for lowest loss model: 19.293417036533356
| Start Iteration 5:
| Iteration 5, Lowest loss: 7.206420330657853
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 1024, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 2048, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 8], 'decoder_ende_attention_heads': [4, 4, 4, 4, 8, 8], 'decoder_arbitrary_ende_attn': [2, 2, 1, 2, 2, 1]}}
| Predicted latency for lowest loss model: 19.293417036533356
| Start Iteration 6:
| Iteration 6, Lowest loss: 7.184933528294303
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 1024, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 8], 'decoder_ende_attention_heads': [4, 4, 4, 4, 4, 8], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, -1]}}
| Predicted latency for lowest loss model: 19.142058491706848
| Start Iteration 7:
| Iteration 7, Lowest loss: 7.183392709565223
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [8, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 3072, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 4, 4, 8], 'decoder_ende_attention_heads': [4, 4, 4, 8, 8, 8], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 1]}}
| Predicted latency for lowest loss model: 19.231361150741577
| Start Iteration 8:
| Iteration 8, Lowest loss: 7.171226473238596
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 1024, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 3072, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 8], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.33850646018982
| Start Iteration 9:
| Iteration 9, Lowest loss: 7.165177479752308
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 3072, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 4, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 8], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.23746168613434
| Start Iteration 10:
| Iteration 10, Lowest loss: 7.164414161286282
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 2048, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 8], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.335204362869263
| Start Iteration 11:
| Iteration 11, Lowest loss: 7.162232388884331
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 4, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 8], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.259604811668396
| Start Iteration 12:
| Iteration 12, Lowest loss: 7.162232388884331
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 4, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 8], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.259604811668396
| Start Iteration 13:
| Iteration 13, Lowest loss: 7.161250393028177
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 2048], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.429346919059753
| Start Iteration 14:
| Iteration 14, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 15:
| Iteration 15, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 16:
| Iteration 16, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 17:
| Iteration 17, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 18:
| Iteration 18, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 19:
| Iteration 19, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 20:
| Iteration 20, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 21:
| Iteration 21, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 22:
| Iteration 22, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 23:
| Iteration 23, Lowest loss: 7.160574420154249
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 2048, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.29745227098465
| Start Iteration 24:
| Iteration 24, Lowest loss: 7.159705452088607
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 4, 4, 4, 8], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.342845678329468
| Start Iteration 25:
| Iteration 25, Lowest loss: 7.159705452088607
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 8]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 4, 4, 4, 8], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.342845678329468
| Start Iteration 26:
| Iteration 26, Lowest loss: 7.158871877574712
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.2182794213295
| Start Iteration 27:
| Iteration 27, Lowest loss: 7.158871877574712
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.2182794213295
| Start Iteration 28:
| Iteration 28, Lowest loss: 7.158871877574712
| Config for lowest loss model: {'encoder': {'encoder_embed_dim': 640, 'encoder_layer_num': 6, 'encoder_ffn_embed_dim': [1024, 1024, 3072, 2048, 3072, 1024], 'encoder_self_attention_heads': [4, 8, 8, 8, 8, 4]}, 'decoder': {'decoder_embed_dim': 640, 'decoder_layer_num': 6, 'decoder_ffn_embed_dim': [1024, 3072, 3072, 3072, 3072, 3072], 'decoder_self_attention_heads': [8, 8, 4, 8, 4, 4], 'decoder_ende_attention_heads': [4, 4, 4, 8, 4, 4], 'decoder_arbitrary_ende_attn': [-1, 2, 2, 2, 2, 2]}}
| Predicted latency for lowest loss model: 19.2182794213295
| Start Iteration 29:
| epoch 013 | valid on 'valid' subset:  11%|████▊                                        | 3/28 [00:08<01:10,  2.81s/it]Traceback (most recent call last):
  File "/home/suhyeon/myHAT/evo_search.py", line 110, in <module>
    cli_main()
  File "/home/suhyeon/myHAT/evo_search.py", line 106, in cli_main
    main(args)
  File "/home/suhyeon/myHAT/evo_search.py", line 51, in main
    best_config = evolver.run_evo_search()
  File "/home/suhyeon/myHAT/fairseq/evolution.py", line 217, in run_evo_search
    popu_scores = self.get_scores(popu)
  File "/home/suhyeon/myHAT/fairseq/evolution.py", line 281, in get_scores
    scores = validate_all(self.args, self.trainer, self.task, self.epoch_iter, configs)
  File "/home/suhyeon/myHAT/fairseq/evolution.py", line 401, in validate_all
    trainer.valid_step(sample)
  File "/home/suhyeon/myHAT/fairseq/trainer.py", line 438, in valid_step
    _loss, sample_size, logging_output = self.task.valid_step(
  File "/home/suhyeon/myHAT/fairseq/tasks/fairseq_task.py", line 241, in valid_step
    loss, sample_size, logging_output = criterion(model, sample)
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/suhyeon/myHAT/fairseq/criterions/label_smoothed_cross_entropy.py", line 56, in forward
    net_output = model(**sample['net_input'])
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/suhyeon/myHAT/fairseq/models/fairseq_model.py", line 223, in forward
    decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/suhyeon/myHAT/fairseq/models/transformer_super.py", line 622, in forward
    x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
  File "/home/suhyeon/myHAT/fairseq/models/transformer_super.py", line 693, in extract_features
    x, attn = layer(
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/suhyeon/myHAT/fairseq/models/transformer_super.py", line 1121, in forward
    x = self.activation_fn(self.fc1(x))
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/suhyeon/miniconda3/envs/SH/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/suhyeon/myHAT/fairseq/modules/linear_super.py", line 58, in forward
    return F.linear(x, self.samples['weight'], self.samples['bias'])
KeyboardInterrupt
```