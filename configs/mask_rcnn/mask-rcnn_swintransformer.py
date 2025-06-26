from myselfsup1.mmselfsup.models.backbones.mask_resnet import MaskResNetV1c
from mmengine import read_base
with read_base():
    from .._base_.models.mask_rcnn_r50_fpn import * # noqa
    from .._base_.datasets.ChestXDet import *  # noqa
    from .._base_.schedules.schedule_300e import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(dict(
    type='MaskRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[133.680, 133.680, 133.680],
        std=[63.986, 63.986, 63.986],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=96,
        patch_size=4,
        # window_size=7,
        window_size=14,
        mlp_ratio=4,
        # depths=(2, 2, 6, 2),
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),  # 根据实际需求选择输出层
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        # drop_path_rate=0.1,
        drop_path_rate=0.3,
        norm_cfg=dict(type='LN', eps=1e-6),
        with_cp=False,
        # pretrained=None,
        pretrained='path_to_pretrained/swin_tiny_patch4_window7.pth'
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],  # 应与Swin Transformer的输出维度匹配
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2,4,8,16],
            ratios=[0.5,1.0,2.0,],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=13,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=13,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
)
)