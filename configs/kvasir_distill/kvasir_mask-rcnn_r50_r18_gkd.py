from mmengine import read_base

from configs._base_.hook.schedule_hook import DistillLossWeightScheduleHook
from razor.models.losses.kvasir_gkd import ChannelWiseDivergenceWithU
from mmdet.engine.hooks.get_epoc_student_hook import SetEpochInfoHook
from mmdet.engine.hooks.schedule__hook import DistillLossWeightScheduleHookV2
with read_base():
    from configs._base_.datasets.Kvasir_SEG import *  # noqa
    from configs._base_.schedules.schedule_300e import *  # noqa
    from configs._base_.default_runtime import *  # noqa

    from configs.mask_rcnn.kvasir_mask_rcnn_r50_fpn import model as teacher  # noqa
    from configs.mask_rcnn.kvasir_mask_rcnn_r18_fpn import model as student  # noqa

teacher_ckpt = '/home/jz207/workspace/liull/MMDetection/Tools/work_dirs/kvasir_mask_rcnn_r50_fpn/t1_epoch_55_940.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=student,
    teacher=teacher,
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            s_feat=dict(type='ModuleOutputs', source='roi_head.mask_head'),
            s_feat1=dict(type='mmrazor.ModuleOutputs', source='roi_head.bbox_head.fc_reg'),
        ),
        teacher_recorders=dict(
            t_feat=dict(type='ModuleOutputs', source='roi_head.mask_head'),
            t_feat1=dict(type='mmrazor.ModuleOutputs', source='roi_head.bbox_head.fc_reg'),
        ),
        distill_losses=dict(
            loss_cwd=dict(
                type=ChannelWiseDivergenceWithU,
                epoch=0,
                gamma=2.0,
                consistency=1.0,
                consistency_rampup=0.0,
                tau=3,
                loss_weight=1.0,
            ),
            loss_reg=dict(
                type='mmdet.SmoothL1Loss',
                loss_weight=1,
            ),
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
            loss_reg=dict(
                pred=dict(
                    from_student=True,
                    recorder='s_feat1',
                    data_idx=None,
                ),
                target=dict(
                    from_student=False,
                    recorder='t_feat1',
                    data_idx=None,
                ),
            ),
        ),

    ))

find_unused_parameters = True
custom_hooks = [dict(type=SetEpochInfoHook)]
# custom_hooks.append(
#     dict(
#         type=DistillLossWeightScheduleHook,
#         # loss_names=['loss_pre', 'loss_post'],
#         loss_names=['loss_post'],
#         eta_min=0.5, gamma=0.5 / 300
#     ))

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')

