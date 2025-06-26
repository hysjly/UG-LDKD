from mmengine.hooks import Hook
from mmengine.logging import MessageHub


class DistillLossWeightScheduleHook(Hook):

    def __init__(self,
                 loss_names,
                 eta_min=0.001,
                 alpha=1.0,
                 gamma=0.001):

        self.loss_names = loss_names if isinstance(loss_names, list) else [loss_names]
        self.eta_min = eta_min  # 损失权重的下限
        self.alpha = alpha  # 初始的损失权重
        self.gamma = gamma  # 每个训练周期减少的权重值
        self.set = False

    def before_train(self, runner) -> None:
        distill_losses = getattr(runner.model.distiller, 'distill_losses')
        message_hub = MessageHub.get_current_instance()
        for loss_name in self.loss_names:
            if f'train/{loss_name}_weight' in message_hub.log_scalars:
                self.alpha = message_hub.get_scalar(f'train/{loss_name}_weight').current()
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                logger.info(f'resumed loss weight: train/{loss_name}_weight, value: {self.alpha}')
            else:
                message_hub.update_scalar(f'train/{loss_name}_weight', self.alpha)
            distill_losses[loss_name].loss_weight = self.alpha

    def after_train_epoch(self, runner) -> None:
        distill_losses = getattr(runner.model.distiller, 'distill_losses')
        message_hub = MessageHub.get_current_instance()
        self.alpha -= self.gamma
        if self.alpha <= self.eta_min:
            self.alpha = self.eta_min
        for loss_name in self.loss_names:
            distill_losses[loss_name].loss_weight = self.alpha
            message_hub.update_scalar(f'train/{loss_name}_weight', self.alpha)
