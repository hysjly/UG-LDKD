import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


class DataSet(torch.utils.data.Dataset):
    def __init__(self, set, dataPath, dataName, width, height):
        super(DataSet, self).__init__()
        self.set = set
        self.path = dataPath
        self.name = dataName
        self.width = width
        self.height = height
        self.transform = transforms.ToTensor()

        # 图像和掩码的路径
        self.image_path = os.path.join(self.path, self.set, 'image')
        self.mask_path = os.path.join(self.path, self.set, 'mask')

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.image_path, self.name[idx])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), Image.NEAREST)
        img = np.array(img)

        # 加载对应的掩码图像
        mask_name = self.name[idx].replace('.jpg', '.jpg')  # 假设掩码图像与原图像同名但带有"_mask"后缀
        mask_path = os.path.join(self.mask_path, mask_name)
        mask = Image.open(mask_path).convert("L")  # 将掩码图像加载为灰度模式
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)

        # 将图像和掩码转换为张量
        img = self.transform(img)
        mask = torch.from_numpy(mask).unsqueeze(0)  # 添加通道维度

        return img, mask
