import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import cv2

class DataSet(torch.utils.data.Dataset):
    def __init__(self, set, dataPath, dataName, jsonPath, width, height):
        super(DataSet, self).__init__()
        self.set = set
        self.path = dataPath
        self.name = dataName
        self.jsonPath = jsonPath
        self.width = width
        self.height = height
        self.transform = transforms.ToTensor()

        # 加载 COCO 数据
        self.coco = COCO(self.jsonPath)
        self.image_ids = self.coco.getImgIds()

        # 创建类别ID映射，将原始ID映射到连续的索引
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.category_count = len(self.cat_ids)  # 用于掩码的通道数

    def __len__(self):
        return len(self.name)

    # def __getitem__(self, idx):
    #     # 加载灰度图像
    #     img_path = self.path + self.set + '/train/' + self.name[idx]
    #     img = Image.open(img_path).convert("L")  # Convert to grayscale
    #     img = img.resize((self.width, self.height), Image.NEAREST)
    #     img = self.transform(img)  # Transform to tensor
    #
    #     # 初始化掩码
    #     mask = np.zeros((self.height, self.width, self.category_count), dtype=np.uint8)
    #
    #     # 获取图像ID及其注释
    #     image_id = self.image_ids[idx]
    #     annotation_ids = self.coco.getAnnIds(imgIds=image_id)
    #     annotations = self.coco.loadAnns(annotation_ids)
    #
    #     # 遍历注释并生成掩码
    #     for annotation in annotations:
    #         original_cat_id = annotation['category_id']
    #         if original_cat_id in self.cat_id_to_index:
    #             mapped_cat_id = self.cat_id_to_index[original_cat_id]
    #             ann_mask = self.coco.annToMask(annotation)
    #             ann_mask = cv2.resize(ann_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
    #             mask[:, :, mapped_cat_id] = np.maximum(mask[:, :, mapped_cat_id], ann_mask)
    #
    #     mask = self.transform(mask)  # Transform mask to tensor
    #     return img, mask
    def __getitem__(self, idx):
        # 加载灰度图像
        img_path = self.path + self.set + '/train/' + self.name[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img = img.resize((self.width, self.height), Image.NEAREST)
        img = self.transform(img)  # Transform to tensor

        # 初始化掩码
        mask = np.zeros((self.height, self.width, self.category_count), dtype=np.uint8)

        # 获取图像ID及其注释
        image_id = self.image_ids[idx]
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # 遍历注释并生成掩码
        for annotation in annotations:
            original_cat_id = annotation['category_id']
            if original_cat_id in self.cat_id_to_index:
                mapped_cat_id = self.cat_id_to_index[original_cat_id]
                segm = annotation['segmentation']

                # 检查segmentation是否为多边形列表格式
                if isinstance(segm, list):
                    for poly in segm:
                        # 将多边形展平
                        if isinstance(poly[0], list):
                            poly = [p for sublist in poly for p in sublist]

                        ann_mask = self.coco.annToMask({
                            'segmentation': [poly],
                            'image_id': image_id,
                            'size': [self.height, self.width]
                        })
                        ann_mask = cv2.resize(ann_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                        mask[:, :, mapped_cat_id] = np.maximum(mask[:, :, mapped_cat_id], ann_mask)
                else:
                    # 直接使用RLE格式的分割
                    ann_mask = self.coco.annToMask(annotation)
                    ann_mask = cv2.resize(ann_mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                    mask[:, :, mapped_cat_id] = np.maximum(mask[:, :, mapped_cat_id], ann_mask)

        # 将mask从 (height, width, category_count) 转换为张量格式 (category_count, height, width)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
        return img, mask
