from mmengine import read_base
from ad_idea.backbone.wrn import WideResNet

with read_base():
    from .._base_.models.faster_rcnn_r50_fpn import *  # noqa
    from .._base_.datasets.DX import *  # noqa
    from .._base_.schedules.schedule_300e import *  # noqa
    from .._base_.default_runtime import *  # noqa

model.update(dict(
    backbone=dict(
        type=WideResNet,
        depth=16,
        widen_factor=2,
        dropout_rate=0.3,
    ),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 128],
        out_channels=256,
        num_outs=5),
))
