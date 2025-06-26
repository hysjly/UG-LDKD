# from mmcv.runner import Hook
import random

from mmengine.hooks import Hook

from configs._base_.datasets.data_agu.class_agu import SpecifiedClassAugmentor


# Example hook to add SpecifiedClassAugmentor to the training loop
class SpecifiedClassAugmentationHook(Hook):
    def __init__(self, specified_class_ids, batch_interval):
        self.specified_class_ids = specified_class_ids
        self.batch_interval = batch_interval

    def before_epoch(self, runner):
        dataset = runner.data_loader.dataset
        augmentor = SpecifiedClassAugmentor(specified_class_ids=self.specified_class_ids,
                                            batch_interval=self.batch_interval)
        augmentor.find_specified_class(dataset)
        runner.data_loader.dataset = AugmentedDataset(runner.data_loader.dataset, augmentor, self.batch_interval)


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
