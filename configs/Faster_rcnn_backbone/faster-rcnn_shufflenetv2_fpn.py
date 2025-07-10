
_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco.py',
    '../_base_/schedules/schedule_300e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='ShuffleNetV2',
        widen_factor=1.0,  # 根据需要选择，例如 0.5, 1.0, 1.5, 2.0
        out_indices=(0, 1, 2),  # 输出的层级，根据需要调整
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=False,
        init_cfg=None  # 如果需要预训练模型，可以指定
    ),
    neck=dict(
        type='FPN',
        in_channels=[116, 232, 464],
        out_channels=256,
        num_outs=5
    ),
)
