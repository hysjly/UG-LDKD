from mmengine.runner.loops import EpochBasedTrainLoop, ValLoop, TestLoop
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler import LinearLR, MultiStepLR
from mmengine.hooks import IterTimerHook, ParamSchedulerHook, DistSamplerSeedHook, EmptyCacheHook, LoggerHook, \
    CheckpointHook
from mmdet.engine.hooks import DetVisualizationHook
from torch.optim import SGD


train_cfg=dict(type=EpochBasedTrainLoop,max_epochs=300, val_interval=1)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# learning rate
param_scheduler = [
    dict(
        type=LinearLR, start_factor=0.001, by_epoch=True, begin=0, end=300),
    dict(
        type=MultiStepLR,
        begin=0,
        end=300,
        by_epoch=True,
        milestones=[50, 100, 200, 250],
        gamma=0.1)
]
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, interval=25, save_best='coco/bbox_mAP_50', rule='greater'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=DetVisualizationHook))
custom_hooks = [
    dict(
        type=EmptyCacheHook,
        before_epoch=False,
        after_epoch=True,
        after_iter=False),
]

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=SGD, lr=0.02, momentum=0.9, weight_decay=0.0001))

auto_scale_lr = dict(enable=False, base_batch_size=16)
