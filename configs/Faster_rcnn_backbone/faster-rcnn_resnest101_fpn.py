from mmengine import read_base
with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import * # noqa
    from .._base_.datasets.DX import *  # noqa
    from .._base_.schedules.schedule_300e import *  # noqa
    from .._base_.default_runtime import *  # noqa
model.update(dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNeSt',
        depth=101,  # 使用 ResNeSt-101
        groups=1,
        base_width=4,
        radix=2,
        reduction_factor=4,
        avg_down_stride=True,
        out_indices=(0, 1, 2, 3),  # 输出索引，对应于不同的阶段
        frozen_stages=-1,  # 如果要冻结某些阶段则设置该参数
        norm_cfg=dict(type='BN', requires_grad=True),  # 使用批量归一化
        style='pytorch'  # 根据 ResNeSt 的实现选择
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # 根据 ResNeSt-101 的输出通道数设置
        out_channels=256,
        num_outs=5
    ),
)
)