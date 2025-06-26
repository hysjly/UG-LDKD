from mmengine import read_base
with read_base():
    from .._base_.models.mask_rcnn_r50_fpn import * # noqa
    from .._base_.datasets.ChestXDet import *  # noqa
    from .._base_.schedules.schedule_300e import *  # noqa
    from .._base_.default_runtime import *  # noqa
model.update(dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=101,  # 使用 ResNeSt-101
        out_indices=(0, 1, 2, 3),  # 输出索引，对应于不同的阶段
        frozen_stages=-1,  # 如果要冻结某些阶段则设置该参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 使用批量归一化
        style='pytorch' , # 根据 ResNeSt 的实现选择
        init_cfg=dict(type='Pretrained',checkpoint='/home/jz207/workspace/liull/MMDetection/mmpretrain/tools/work_dirs/byol_resnet101_ChestXDet/backbone.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # 根据 ResNeSt-101 的输出通道数设置
        out_channels=256,
        num_outs=5
    ),
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