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
        type='ResNeSt',
        depth=50,  # 可以选择 50, 101, 152 等
        groups=1,
        base_width=4,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        out_indices=(0, 1, 2, 3),  # 根据需要选择输出层
        frozen_stages=-1,  # 如果需要冻结某些阶段，设置这个参数
        norm_cfg=dict(type='BN', requires_grad=True)  # 使用批量归一化
        # init_cfg=dict(type='Pretrained',checkpoint='torchvision://resnet50'),
        # dcn=dict(type='DCNv2',deform_groups=1,fallback_on_stride=False),
        # stage_with_dcn=(False,True,True,True),
        # # init_cfg=dict(type='Pretrained',checkpoint='/home/jz207/workspace/liull/MMDetection/myselfsup1/tools/work_dirs/configs/mae_reset50/mae2_epoch_297_0620.pth',prefix='backbone.')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # 这需要根据选择的 ResNeSt 配置
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