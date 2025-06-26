import random
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from PIL import Image

@TRANSFORMS.register_module()
class RandomCropWithScaling(BaseTransform):
    """随机裁剪并缩放包含目标的图像区域，以增加数据集的多样性

    Args:
        crop_size (tuple): 裁剪后图像的目标尺寸 (width, height)。
        ratio_range (tuple, optional): 随机选择裁剪区域的比例范围。默认值为 (0.5, 2.0)。
        prob (float, optional): 应用此变换的概率。默认值为 0.5。
    """

    def __init__(self, crop_size, ratio_range=(0.5, 2.0), prob=0.5):
        self.crop_size = crop_size
        self.ratio_range = ratio_range
        self.prob = prob

    def transform(self, results):
        if random.random() > self.prob:
            return results

        img = results['img']
        img_h, img_w = img.shape[:2]

        # 检查是否存在有效的 ground truth bbox
        if len(results['gt_bboxes']) == 0:
            return results  # 如果没有 ground truth，跳过增强

        # 获取 ground truth 边界框
        gt_bbox = random.choice(results['gt_bboxes'])

        # 检查 gt_bbox 的长度是否为 4（确保它是 [x_min, y_min, x_max, y_max] 格式）
        if len(gt_bbox) != 4:
            return results  # 如果不是有效的边界框，跳过增强

        # 获取 ground truth 的中心点和宽高
        x_center = (gt_bbox[0] + gt_bbox[2]) / 2
        y_center = (gt_bbox[1] + gt_bbox[3]) / 2
        bbox_w = gt_bbox[2] - gt_bbox[0]
        bbox_h = gt_bbox[3] - gt_bbox[1]

        # 计算裁剪框的宽高
        ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
        crop_w = int(bbox_w * ratio)
        crop_h = int(bbox_h * ratio)

        # 计算裁剪框的左上角坐标，确保在图像边界内
        x1 = max(0, int(x_center - crop_w / 2))
        y1 = max(0, int(y_center - crop_h / 2))
        x2 = min(img_w, x1 + crop_w)
        y2 = min(img_h, y1 + crop_h)

        # 裁剪图像
        cropped_img = img[y1:y2, x1:x2]

        # 调整裁剪后的图像大小为目标 crop_size
        cropped_img = np.array(
            Image.fromarray(cropped_img).resize(self.crop_size, Image.BILINEAR))

        # 调整 ground truth 坐标
        new_gt_bboxes = []
        for bbox in results['gt_bboxes']:
            # 检查 bbox 是否有效
            if len(bbox) != 4:
                continue

            # 计算在裁剪框中的相对坐标
            new_x1 = max(0, bbox[0] - x1)
            new_y1 = max(0, bbox[1] - y1)
            new_x2 = min(crop_w, bbox[2] - x1)
            new_y2 = min(crop_h, bbox[3] - y1)

            # 重新缩放到目标尺寸
            new_x1 = new_x1 * self.crop_size[0] / crop_w
            new_y1 = new_y1 * self.crop_size[1] / crop_h
            new_x2 = new_x2 * self.crop_size[0] / crop_w
            new_y2 = new_y2 * self.crop_size[1] / crop_h

            new_gt_bboxes.append([new_x1, new_y1, new_x2, new_y2])

        # 更新 results 字典
        results['img'] = cropped_img
        results['img_shape'] = cropped_img.shape
        results['gt_bboxes'] = np.array(new_gt_bboxes, dtype=np.float32)

        return results
    #
    # dict(type=RandomCropWithScaling,
    #      crop_size=(512,512),
    #      ratio_range=(0.5,2.0),
    #      prob=0.5),
    # #dict(type='GridMask',use_h=True,use_w=True,rotate=1,offset=False,ratio=0.5,mode=0,prob=0.5),#网格掩码
    # dict(type='PhotoMetricDistortion',brightness_delta=32,contrast_range=(0.5,1.5),saturation_range=(0.5,1.5),hue_delta=18),#亮度变换

