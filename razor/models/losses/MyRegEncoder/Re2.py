import torch
import torch.nn.functional as F
from ....registry import MODELS

@MODELS.register_module()
class MaskResizer:
    def __init__(self, image_size=(1024, 1024)):
        """
        初始化 MaskResizer 模块，用于将 MaskHead 输出的掩膜还原为原始图像大小。

        参数：
        - image_size: tuple，表示原始图像的大小 (h, w)。
        """
        self.image_size = image_size

    def convert_to_full_size_mask(self, input_tensor, rois):
        """
        将 MaskHead 输出的 Tensor (N, C, H, W) 转换为原始图像大小的掩膜 Tensor (C, h, w)。

        参数：
        - input_tensor: Tensor of shape (N, C, H, W)，表示感兴趣区域（ROI）的掩膜预测。
        - rois: Tensor of shape (N, 5)，表示每个 ROI 的坐标 (batch_idx, x1, y1, x2, y2)。

        返回：
        - full_size_mask: Tensor of shape (C, h, w)，表示原始图像大小的掩膜。
        """
        N, C, H, W = input_tensor.shape
        h, w = self.image_size

        # 初始化一个全零的掩码图像，大小为 (C, h, w)
        full_size_mask = torch.zeros((C, h, w), dtype=torch.float32, device=input_tensor.device)

        # 遍历每个 ROI，进行掩膜插值并合并到完整图像中
        for i in range(N):
            # 获取当前 ROI 的坐标 (batch_index, x1, y1, x2, y2)
            batch_index, x1, y1, x2, y2 = rois[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 检查 ROI 的宽度和高度是否大于 0
            if x2 > x1 and y2 > y1:
                # 获取当前 ROI 的预测掩膜，形状为 (C, H, W)
                mask = input_tensor[i]

                # 将掩膜插值到 ROI 的大小 (y2 - y1, x2 - x1)
                resized_mask = F.interpolate(mask.unsqueeze(0), size=(y2 - y1, x2 - x1), mode='bilinear', align_corners=False)
                resized_mask = resized_mask.squeeze(0)  # Shape: (C, y2 - y1, x2 - x1)

                # 将插值后的掩膜映射到原始图像中的对应位置
                # 由于 full_size_mask 是多通道的，我们需要按类别进行更新
                full_size_mask[:, y1:y2, x1:x2] = torch.max(full_size_mask[:, y1:y2, x1:x2], resized_mask)

        return full_size_mask
