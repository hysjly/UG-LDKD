import torch
import torch.nn.functional as F
from ....registry import MODELS

# @MODELS.register_module()
# class MaskResizer:
#     def __init__(self, image_size=(512, 512)):
#         """
#         初始化 MaskResizer 模块，用于将 MaskHead 输出的掩膜还原为原始图像大小。
#
#         参数：
#         - image_size: tuple(h, w)，表示原始图像的大小。
#         """
#         self.image_size = image_size
#
#     def convert_to_full_size_mask(self, input_tensor: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
#         """
#         将 (N, C, H, W) 的ROI掩膜还原到原始图像尺度上。
#
#         参数：
#         - input_tensor: (N, C, H, W) 的张量，每个ROI对应一块掩膜预测。
#         - rois: (N, 5) 的张量，每行为 [batch_idx, x1, y1, x2, y2]。
#
#         返回：
#         - full_size_mask: (B, C, h, w) 的张量，其中 B 为图像数量 (max(batch_idx)+1)。
#         """
#         N, C, H, W = input_tensor.shape
#         h, w = self.image_size
#
#         batch_idxs = rois[:, 0].long()
#         num_images = batch_idxs.max().item() + 1
#
#         # 初始化全图掩膜
#         full_size_mask = torch.zeros((num_images, C, h, w), dtype=torch.float32, device=input_tensor.device)
#
#         for i in range(N):
#             batch_idx = int(rois[i, 0].item())
#             x1 = int(round(rois[i, 1].item()))
#             y1 = int(round(rois[i, 2].item()))
#             x2 = int(round(rois[i, 3].item()))
#             y2 = int(round(rois[i, 4].item()))
#
#             # 确保ROI在图像范围内
#             x1 = max(0, min(x1, w))
#             x2 = max(0, min(x2, w))
#             y1 = max(0, min(y1, h))
#             y2 = max(0, min(y2, h))
#
#             roi_height = y2 - y1
#             roi_width = x2 - x1
#
#             # 如果ROI无效或大小为0，跳过
#             if roi_height <= 0 or roi_width <= 0:
#                 continue
#
#             # 原始ROI掩膜
#             mask = input_tensor[i]  # (C, H, W)
#
#             # 使用F.interpolate将ROI内的掩膜缩放到对应区域大小
#             # (C, H, W) -> (1, C, H, W)
#             resized_mask = F.interpolate(
#                 mask.unsqueeze(0),
#                 size=(roi_height, roi_width),
#                 mode='bilinear',
#                 align_corners=False
#             ).squeeze(0)  # (C, roi_height, roi_width)
#
#             # 调试信息(可根据需要取消注释)
#             # print(f"roi_height={roi_height}, roi_width={roi_width}")
#             # print(f"resized_mask.shape={resized_mask.shape}")
#             # print(f"full_size_mask[batch_idx, :, y1:y2, x1:x2].shape={full_size_mask[batch_idx, :, y1:y2, x1:x2].shape}")
#
#             full_slice = full_size_mask[batch_idx, :, y1:y2, x1:x2]
#
#             # 若形状不匹配，可打印调试
#             if full_slice.shape != resized_mask.shape:
#                 print(f"Shape mismatch: full_slice: {full_slice.shape}, resized_mask: {resized_mask.shape}")
#                 # 可以选择 continue 或者尝试修正
#                 continue
#
#             # 使用max进行融合，避免重叠ROI的覆盖问题，也可根据需求改为加和或其他方式
#             full_size_mask[batch_idx, :, y1:y2, x1:x2] = torch.max(full_slice, resized_mask)
#
#         return full_size_mask
import torch
import torch.nn.functional as F
from ....registry import MODELS

@MODELS.register_module()
class MaskResizer:
    def __init__(self, image_size=(1024, 1024)):
        self.image_size = image_size

    def convert_to_full_size_mask(self, input_tensor: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        N, C, H, W = input_tensor.shape
        h, w = self.image_size

        # 若没有roi，直接返回空张量
        if rois.numel() == 0:
            # 根据业务需求返回(1, C, h, w)空张量
            return torch.zeros((1, C, h, w), dtype=torch.float32, device=input_tensor.device)

        batch_idxs = rois[:, 0].long()
        if batch_idxs.numel() == 0:
            # 同理若batch_idxs为空，返回空
            return torch.zeros((1, C, h, w), dtype=torch.float32, device=input_tensor.device)

        num_images = batch_idxs.max().item() + 1

        # full_size_mask不需要梯度
        full_size_mask = torch.zeros((num_images, C, h, w),
                                     dtype=torch.float32,
                                     device=input_tensor.device,
                                     requires_grad=False)

        for i in range(N):
            batch_idx = int(rois[i, 0].item())
            x1 = int(round(rois[i, 1].item()))
            y1 = int(round(rois[i, 2].item()))
            x2 = int(round(rois[i, 3].item()))
            y2 = int(round(rois[i, 4].item()))

            # 保证ROI坐标在图像范围内
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

            roi_height = y2 - y1
            roi_width = x2 - x1

            if roi_height <= 0 or roi_width <= 0:
                # 无效ROI，跳过
                continue

            mask = input_tensor[i]  # (C, H, W)

            resized_mask = F.interpolate(
                mask.unsqueeze(0),
                size=(roi_height, roi_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (C, roi_height, roi_width)

            # 使用no_grad确保原地操作不影响梯度计算
            with torch.no_grad():
                full_slice = full_size_mask[batch_idx, :, y1:y2, x1:x2]
                # 假设resized_mask与full_slice尺寸匹配，如有不匹配需提前检查
                temp = torch.max(full_slice, resized_mask)
                full_slice.copy_(temp)

        return full_size_mask