from torch.utils.data import Sampler
import random
from mmdet.registry import DATA_SAMPLERS
@DATA_SAMPLERS.register_module()
class PercentSampler(Sampler):
    def __init__(self, dataset, percentage=0.2, shuffle=True, seed=None):
        self.dataset = dataset
        self.percentage = percentage
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.seed = seed

    def __iter__(self):
        # 设置随机种子
        if self.seed is not None:
            random.seed(self.seed)

        num_samples = int(len(self.dataset) * self.percentage)
        if self.shuffle:
            sampled_indices = random.sample(self.indices, num_samples)
        else:
            sampled_indices = self.indices[:num_samples]
        return iter(sampled_indices)

    def __len__(self):
        return int(len(self.dataset) * self.percentage)
