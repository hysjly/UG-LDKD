# # Copyright (c) OpenMMLab. All rights reserved.
# from typing import Optional, List, Union, Any
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
#
# from ...registry import MODELS
# from .utils import weighted_loss
# from .MyRegEncoder.Re import MaskResizer
#
#
# @weighted_loss
# def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
#     """A Wrapper of L1 loss.
#     Args:
#         pred (Tensor): The prediction.
#         target (Tensor): The learning target of the prediction.
#
#     Returns:
#         Tensor: loss Tensor
#     """
#     return F.l1_loss(pred, target, reduction='none')
#
#
# def align(s, t):
#     s_channels = s.shape[1]
#     c = t.shape[1]
#     h = t.shape[2]
#     w = t.shape[3]
#     adjust_channels = nn.Sequential(
#         nn.Conv2d(s_channels, c, kernel_size=1, stride=1, padding=0),
#         nn.BatchNorm2d(c),
#         nn.ReLU(inplace=True)).to("cuda")
#     # 2. 调整学生特征的空间尺寸以匹配教师特征
#     upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
#
#     s = adjust_channels(s)
#     s = upsample(s)
#     return s
#
# @MODELS.register_module()
# class LastPreLoss(nn.Module):
#     """Pre-distillation Loss for SingleTeacherDistill.
#     Args:
#         reduction (str, optional): The method that reduces the loss to a
#             scalar. Options are "none", "mean" and "sum".
#         loss_weight (float, optional): The weight of the loss. Defaults to 1.0
#     """
#
#     def __init__(self,
#                  reduction: str = 'mean',
#                  loss_weight: float = 1.0) -> None:
#         super().__init__()
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.mask_resizer = MaskResizer(image_size=(1024, 1024))
#
#     def forward(self,
#                 s_input: Tensor,
#                 s_rois: Tensor,
#                 t_input: Tensor,
#                 t_rois: Tensor,
#                 weight: Optional[Tensor] = None,
#                 avg_factor: Optional[int] = None,
#                 reduction_override: Optional[str] = None) -> List[Union[float, Any]]:
#         # 使用 MaskResizer 将输入还原为原始图像大小
#         s_input = self.mask_resizer.convert_to_full_size_mask(s_input, s_rois)
#         t_input = self.mask_resizer.convert_to_full_size_mask(t_input, t_rois)
#
#         # 对学生特征进行对齐，以便与教师特征对比
#         s_aligned = align(s_input, t_input)
#
#         # 计算学生特征和教师特征的绝对差值
#         absolute_difference = torch.abs(s_aligned - t_input)
#
#         # 直接计算绝对差值的 L1 损失
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss = self.loss_weight * l1_loss(
#             absolute_difference, torch.zeros_like(absolute_difference), weight, reduction=reduction, avg_factor=avg_factor)
#         return loss
