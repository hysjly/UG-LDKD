from backbone.transmed import TransMED
from mmengine import read_base

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa
    from .._base_.datasets.DX import *  # noqa
    from .._base_.schedules.schedule_300e import *  # noqa
    from .._base_.default_runtime import *  # noqa
model.update(dict(
    backbone=dict(
        type=TransMED,
        # img_size=640,
        patch_size=16,
        # in_channels=1,
        embed_dims=512,
        depth=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        out_indices=[0, 1, 2, 3]
    ),
    neck=dict(
        type='FPN',
        in_channels=[512,512,512,512],  # 注意这需要根据 MobileNetV2 的 out_indices 调整
        out_channels=256,
        num_outs=5),))
