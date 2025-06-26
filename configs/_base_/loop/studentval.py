from typing import Dict, List, Sequence, Union

import torch
from mmengine.evaluator import Evaluator
from mmengine.runner import ValLoop, autocast
from torch.utils.data import DataLoader


class StudentModelValLoop(ValLoop):
    """
    Validation loop for the student model in MMrazor distillation framework.
    """
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        if self.runner.distributed:
            assert hasattr(self.runner.model.module, 'student')
            data_preprocessor = self.runner.model.module.data_preprocessor
            self.student_model = self.runner.model.module.student
            self.student_model.data_preprocessor = data_preprocessor
        else:
            assert hasattr(self.runner.model, 'student')
            data_preprocessor = self.runner.model.data_preprocessor
            self.student_model = self.runner.model.student
            self.student_model.data_preprocessor = data_preprocessor

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')

        # Validate student model
        self.student_model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        # Call hooks after validation
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch for student model.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)

        with autocast(enabled=self.fp16):
            # outputs should be sequence of BaseDataElement
            outputs = self.student_model.val_step(data_batch)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)