from myselfsup1.mmselfsup.models.backbones.mask_resnet import MaskResNetV1c
from configs._base_.models.heads.myroi_head import StandardRoIHeadv2
from configs._base_.models.heads.my_maskhead import FCNMaskHeadv1
from mmengine import read_base
with read_base():
    from .._base_.models.mask_rcnn_r50_fpn import * # noqa
    from .._base_.datasets.Kvasir_SEG import *  # noqa
    from .._base_.schedules.schedule_300e import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(dict(
    type='MaskRCNN',
    backbone=dict(
        type='EfficientNet',
        arch='es',  # 使用 EfficientNet-EdgeTPU Small
        out_indices=(2, 3, 4, 5),  # 输出索引，对应于EfficientNet-ES中不同的阶段
        frozen_stages=0,  # 如果要冻结某些阶段则设置该参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 使用批量归一化
        norm_eval=False,  # 训练时是否冻结BN层
        with_cp=False  # 是否使用 checkpoint (梯度检查点)
    ),
    neck=dict(
        type='FPN',
        in_channels=[32, 48, 144, 192],  # 根据EfficientNet-ES的特定层输出通道数设置
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
        # type=StandardRoIHeadv2,
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
            num_classes=1,
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
            # type=FCNMaskHeadv1,
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
)
)