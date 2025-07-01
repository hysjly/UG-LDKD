import numpy as np
import os
import torch
import random
import math
import json
from pycocotools.coco import COCO
def DeviceInitialization(GPUNum):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = torch.device(GPUNum)
    else:
        device = torch.device('cpu')

    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    return device


def DataReading(data_path, set, fracTrain):
    trainIdx = []
    testIdx = []

    # 加载 COCO JSON 数据
    json_path = os.path.join(data_path, 'ChestXDet',  'train.json')
    with open(json_path) as f:
        coco_data = json.load(f)

    # 获取图像文件名
    image_ids = [image['file_name'] for image in coco_data['images']]
    total_images = len(image_ids)

    # 打乱图像顺序
    shuffleIdx = np.arange(total_images)
    shuffleRng = np.random.RandomState(2021)
    shuffleRng.shuffle(shuffleIdx)
    image_ids = np.array(image_ids)[shuffleIdx]

    # 计算训练和测试集数量
    TrainNum = math.ceil(fracTrain * total_images / 100)
    train = image_ids[:TrainNum]
    test = image_ids[TrainNum:]

    # 将文件名添加到索引列表中
    trainIdx.extend(train)
    testIdx.extend(test)

    return trainIdx, testIdx  # 不调用 tolist()






# # 示例用法
# data_path = 'E:\\professional_tool\\project\\PyCharmProject\\MMDetection\\data\\'
# set = 'coco'
# fracTrain = 80  # 比如训练集占 80%
# TrainIdx, TestIdx = DataReading(data_path, set, fracTrain)
#
# print("Training Images:", TrainIdx)
# print("Testing Images:", TestIdx)


class Sampler:
    def __init__(self, device, model, img_shape, sample_size, max_len, step, lr):
        super().__init__()
        self.device = device
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.step = step
        self.lr = lr
        self.examples = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size.item())]

    def sample_new_exmps(self):
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size.item(), 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size.item() - n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(self.device)
        inp_imgs = Sampler.generate_samples(self.model.to(self.device), inp_imgs, steps=self.step, step_size=self.lr)
        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device('cpu')).chunk(self.sample_size.item(), dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    def generate_samples(model, inp_imgs, steps, step_size):
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)
        for idx in range(steps):
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data + noise.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

        inp_imgs.requires_grad = False
        inp_imgs.detach()
        for p in model.parameters():
            p.requires_grad = True

        return inp_imgs


def Sampling(model, inp_imgs, step, step_size):
    inp_imgs.requires_grad = True
    for idx in range(step):
        inp_imgs.data.clamp_(min=-1.0, max=1.0)
        out_imgs = -model(inp_imgs)
        out_imgs.sum().backward()
        inp_imgs.grad.data.clamp_(-0.1, 0.1)
        inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
        inp_imgs.grad.detach_()
        inp_imgs.grad.zero_()
        inp_imgs.data.clamp_(min=-1.0, max=1.0)
    inp_imgs.requires_grad = False
    inp_imgs.detach()

    return inp_imgs
