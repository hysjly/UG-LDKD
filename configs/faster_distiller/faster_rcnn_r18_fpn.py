from myselfsup1.mmselfsup.models.backbones.mask_resnet import MaskResNetV1c
from configs._base_.models.heads.myroi_head import StandardRoIHeadv2
from configs._base_.models.heads.my_maskhead import FCNMaskHeadv1
from mmengine import read_base
with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import * # noqa
    from .._base_.datasets.ChestXDet import *  # noqa
    from .._base_.schedules.schedule_300e import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[133.680, 133.680, 133.680],
        std=[63.986, 63.986, 63.986],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=18,  # 使用 ResNeSt-101
        out_indices=(0, 1, 2, 3),  # 输出索引，对应于不同的阶段
        frozen_stages=1,  # 如果要冻结某些阶段则设置该参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 使用批量归一化
        style='pytorch' , # 根据 ResNeSt 的实现选择
        init_cfg=dict(type='Pretrained',checkpoint='torchvision://resnet18'),
    ),
    neck=dict(in_channels=[64, 128, 256, 512]),
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
            # nms=dict(type='nms', iou_threshold=0.5),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
),
)
)