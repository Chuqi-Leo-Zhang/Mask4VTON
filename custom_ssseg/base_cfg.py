'''SEGMENTOR_CFG for Mask2Former'''
SEGMENTOR_CFG = {
    'type': 'Mask2Former',
    'num_classes': -1,
    'benchmark': True,
    'align_corners': False,
    'work_dir': 'ckpts',
    'eval_interval_epochs': 10,
    'save_interval_epochs': 3,
    'logger_handle_cfg': {'type': 'LocalLoggerHandle', 'logfilepath': ''},
    'training_logging_manager_cfg': {'log_interval_iters': 200},
    'norm_cfg': {'type': 'SyncBatchNorm'},
    'act_cfg': {'type': 'ReLU', 'inplace': True},
    'backbone': {
        'type': 'SwinTransformer', 'structure_type': 'swin_base_patch4_window12_384_22k', 'pretrained': True,
        'selected_indices': (0, 1, 2, 3), 'norm_cfg': {'type': 'LayerNorm'},
        'pretrain_img_size': 384, 'in_channels': 3, 'embed_dims': 128, 'patch_size': 4, 'window_size': 12, 'mlp_ratio': 4,
        'depths': [2, 2, 18, 2], 'num_heads': [4, 8, 16, 32], 'qkv_bias': True, 'qk_scale': None, 'patch_norm': True,
        'drop_rate': 0., 'attn_drop_rate': 0., 'drop_path_rate': 0.3, 'use_abs_pos_embed': False,
    },
    'head': {
        'deep_supervision': True,
        'pixel_decoder': {
            'common_stride': 4, 'transformer_dropout': 0.0, 'transformer_nheads': 8, 'transformer_dim_feedforward': 1024, 
            'transformer_enc_layers': 6, 'conv_dim': 256, 'mask_dim': 256, 'transformer_in_features': ['res3', 'res4', 'res5'], 
            'norm_cfg': {'type': 'GroupNorm', 'num_groups': 32}, 'act_cfg': {'type': 'ReLU', 'inplace': True},
            'input_shape': {'in_channels': [128, 256, 512, 1024], 'strides': [4, 8, 16, 32]},
        },
        'predictor': {
            'in_channels': 256, 'hidden_dim': 256, 'num_queries': 100, 'nheads': 8, 'dim_feedforward': 2048, 'dec_layers': 10, 
            'pre_norm': False, 'mask_dim': 256, 'enforce_input_project': False, 'mask_classification': True,
        },
        'matcher': {
            'num_points': 12544, 'cost_class': 2.0, 'cost_mask': 5.0, 'cost_dice': 5.0,
        },
        'criterion': {
            'pixelsampler_cfg': {'type': 'UGSampler', 'oversample_ratio': 3.0, 'importance_sample_ratio': 0.75, 'num_points': 12544, 'reformat_target': False},
            'eos_coef': 0.1, 'losses': ['labels', 'masks'], 
        },
    },
    'auxiliary': None,
    'losses': {},
    'inference': {
        'forward': {'mode': 'whole', 'cropsize': None, 'stride': None},
        'tta': {'multiscale': [1], 'flip': False, 'use_probs_before_resize': False},
        'evaluate': {'metric_list': ['iou', 'miou']},
    },
    'scheduler': {
        'type': 'PolyScheduler', 'max_epochs': 0, 'power': 0.9,
        'optimizer': {
            'type': 'AdamW', 'lr': 0.0001, 'betas': (0.9, 0.999), 'weight_decay': 0.05, 'eps': 1e-8,
            'params_rules': {
                'base_setting': dict(norm_wd_multiplier=0.0),
                'backbone_net': dict(lr_multiplier=0.1, wd_multiplier=1.0),
                'backbone_net.patch_embed.norm': dict(lr_multiplier=0.1, wd_multiplier=0.0),
                'backbone_net.norm': dict(lr_multiplier=0.1, wd_multiplier=0.0),
                'absolute_pos_embed': dict(lr_multiplier=0.1, wd_multiplier=0.0),
                'relative_position_bias_table': dict(lr_multiplier=0.1, wd_multiplier=0.0),
                'query_embed': dict(lr_multiplier=1.0, wd_multiplier=0.0),
                'query_feat': dict(lr_multiplier=1.0, wd_multiplier=0.0),
                'level_embed': dict(lr_multiplier=1.0, wd_multiplier=0.0),
            },
        }
    },
    'dataset': None,
    'dataloader': None,
}
for stage_id, num_blocks in enumerate(SEGMENTOR_CFG['backbone']['depths']):
    for block_id in range(num_blocks):
        SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'].update({
            f'backbone_net.stages.{stage_id}.blocks.{block_id}.norm': dict(lr_multiplier=0.1, wd_multiplier=0.0)
        })
for stage_id in range(len(SEGMENTOR_CFG['backbone']['depths']) - 1):
    SEGMENTOR_CFG['scheduler']['optimizer']['params_rules'].update({
        f'backbone_net.stages.{stage_id}.downsample.norm': dict(lr_multiplier=0.1, wd_multiplier=0.0)
    })