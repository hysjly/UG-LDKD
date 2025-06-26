from mmengine import read_base

from configs._base_.hook.schedule_hook import DistillLossWeightScheduleHook
from razor.models.losses.sinkd.distkd import DISTLoss
from mmdet.engine.hooks.get_epoc_student_hook import SetEpochInfoHook
from mmdet.engine.hooks.schedule__hook import DistillLossWeightScheduleHookV2
with read_base():
    from configs._base_.datasets.Kvasir_SEG import *  # noqa
    from configs._base_.schedules.schedule_300e import *  # noqa
    from configs._base_.default_runtime import *  # noqa

    from configs.Faster_rcnn.sinkd_faster_rcnn_pvt_fpn import model as teacher  # noqa
    from configs.Faster_rcnn.sinkd_faster_rcnn_r18_fpn import model as student  # noqa

teacher_ckpt = '/home/jz207/workspace/liull/MMDetection/t1_sc4816epoch_43_939.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=student,
    teacher=teacher,
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            s_feat=dict(type='ModuleOutputs', source='roi_head.bbox_head.fc_cls'),
        ),
        teacher_recorders=dict(
            t_feat=dict(type='ModuleOutputs', source='roi_head.bbox_head.fc_cls'),
        ),
        distill_losses=dict(
            loss_cwd=dict(
                type=DISTLoss,              ),
        ),
        loss_forward_mappings=dict(
            loss_cwd=dict(
                s_feat=dict(
                    from_student=True,
                    recorder='s_feat',
                    data_idx=None,
                ),
                t_feat=dict(
                    from_student=False,
                    recorder='t_feat',
                    data_idx=None,
                ),
            ),
        ),

    ))

find_unused_parameters = True
custom_hooks = [dict(type=SetEpochInfoHook)]

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')

