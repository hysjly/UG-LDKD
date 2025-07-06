# Copyright (c) OpenMMLab. All rights reserved.
from albumentations import RandomBrightnessContrast
from mmdet.datasets import CocoDataset
from configs._base_.datasets.RandomCropWithScaling import RandomCropWithScaling
# dataset settings

dataset_type = CocoDataset
data_root = '/home/jz207/workspace/liull/MMDetection/data/ChestXDet/'

backend_args = None

DX_METAINFO = dict(
    classes=('Cardiomegaly', 'Nodule', 'Consolidation', 'Effusion', 'Pleural Thickening',
             'Fibrosis', 'Emphysema', 'Calcification','Atelectasis','Pneumothorax','Fracture','Mass','Diffuse Nodule')
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True,poly2mask=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='RandomRotate',degree=20),
    # dict(type='CutMix',cutmix_prob=0.5),
    # dict(type='MixUp'),
    # dict(type=RandomCropWithScaling,
    #      crop_size=(512,512),
    #      ratio_range=(0.5,2.0),
    #      prob=0.5),
    #dict(type='GridMask',use_h=True,use_w=True,rotate=1,offset=False,ratio=0.5,mode=0,prob=0.5),#网格掩码
    # dict(type='PhotoMetricDistortion',brightness_delta=32,contrast_range=(0.5,1.5),saturation_range=(0.5,1.5),hue_delta=18),#亮度变换
    dict(
        type='Albu',
        transforms=[
            dict(type='Blur', p=0.01),
            dict(type='MedianBlur', p=0.01),
            dict(type='ToGray', p=0.01),
            dict(type='CLAHE', p=0.01)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes',
        }),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='Pad',size_divisor=32),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True,with_mask=True,poly2mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    dataset=dict(
        _scope_='mmdet',
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/ChestX_Det_train_coco.json',
        data_prefix=dict(img='train/'),
        # filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args,
        metainfo=DX_METAINFO
    )
)
val_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _scope_='mmdet',
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/ChestX_Det_test_coco_fixed.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=DX_METAINFO))

val_evaluator = dict(
    _scope_='mmdet',
    type='CocoMetric',
    ann_file=data_root + 'annotations/ChestX_Det_test_coco_fixed.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    classwise=True)

test_dataloader = dict(
    batch_size=4,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _scope_='mmdet',
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/ChestX_Det_test_coco_fixed.json',
        data_prefix=dict(img='test/'),
        # ann_file='annotations/ChestX_Det_train_coco.json',
        # data_prefix=dict(img='train/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
        metainfo=DX_METAINFO))

test_evaluator = dict(
    _scope_='mmdet',
    type='CocoMetric',
    outfile_prefix=data_root,
    ann_file=data_root + 'annotations/ChestX_Det_test_coco_fixed.json',
    # ann_file=data_root + 'annotations/ChestX_Det_train_coco.json',
    metric='bbox',
    format_only=True,
    backend_args=backend_args, classwise=True)
