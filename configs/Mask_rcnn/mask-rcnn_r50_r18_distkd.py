from mmengine import read_base


from razor.models.losses.distkd import DISTLoss
from mmdet.engine.hooks.get_epoc_student_hook import SetEpochInfoHook

with read_base():
    from configs._base_.datasets.ChestXDet import *  # noqa
    from configs._base_.schedules.schedule_300e import *  # noqa
    from configs._base_.default_runtime import *  # noqa

    from configs.mask_rcnn.mask_rcnn_r50_fpn import model as teacher  # noqa
    from configs.mask_rcnn.mask_rcnn_r18_fpn import model as student  # noqa

teacher_ckpt = '/home/jz207/workspace/liull/MMDetection/t1_epoch_53_418.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=student,
    teacher=teacher,
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            s_roi_cls=dict(type='ModuleOutputs', source='roi_head.mask_head'),
            s_feat1=dict(type='mmrazor.ModuleOutputs', source='roi_head.bbox_head.fc_reg'),

        ),
        teacher_recorders=dict(
            t_roi_cls=dict(type='ModuleOutputs', source='roi_head.mask_head'),
            t_feat1=dict(type='mmrazor.ModuleOutputs', source='roi_head.bbox_head.fc_reg'),

        ),
        distill_losses=dict(
            loss_roi_cls=dict(type=DISTLoss, tau=4, inter_loss_weight=1.0,intra_loss_weight=2.0),
            loss_reg=dict(
                type='mmdet.SmoothL1Loss',
                loss_weight=1,
            ),

        ),
        loss_forward_mappings=dict(
            loss_roi_cls=dict(
                s_feat=dict(from_student=True, recorder='s_roi_cls'),
                t_feat=dict(from_student=False, recorder='t_roi_cls')
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

val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
# train_cfg = dict(type=EpochBasedTrainLoop, max_epochs=300, val_interval=1)
# test_cfg = dict(type=TestLoop)