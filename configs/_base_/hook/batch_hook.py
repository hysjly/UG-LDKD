import random

from mmengine.hooks import Hook
from configs._base_.datasets.data_agu.class_agu import SpecifiedClassAugmentor


class EveryTenBatchHook(Hook):
    def __init__(self, specified_class_ids, batch_interval=10):
        self.specified_class_ids = specified_class_ids
        self.batch_interval = batch_interval
        self.batch_count = 0

    def after_train_iter(self, runner, **kwargs):
        # This method is called after every training iteration (batch)
        self.batch_count += 1

        # Perform the desired operation every 'batch_interval' batches
        if self.batch_count % self.batch_interval == 0:
            # Add your desired operation here
            self.perform_operation(runner)

    def perform_operation(self, runner):
        dataset = runner.data_loader.dataset
        augmentor = SpecifiedClassAugmentor(specified_class_ids=self.specified_class_ids,
                                            batch_interval=self.batch_interval)
        augmentor.find_specified_class(dataset)
        runner.data_loader.dataset = AugmentedDataset(runner.data_loader.dataset, augmentor, self.batch_interval)

    def before_epoch(self, runner):
        # Reset batch count at the start of each epoch
        self.batch_count = 0


class AugmentedDataset:
    def __init__(self, dataset, augmentor, batch_interval):
        self.dataset = dataset
        self.augmentor = augmentor
        self.batch_interval = batch_interval

    def __getitem__(self, idx):
        results = self.dataset[idx]
        if random.randint(0, self.batch_interval) == 0:  # Every 10 batches
            results = self.augmentor.augment(results)
        return results

    def __len__(self):
        return len(self.dataset)
