# from typing import Optional, List, Union, Any
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor
#
# from ...registry import MODELS
# from .utils import weighted_loss
# from .MyRegEncoder.RegEncoderv2 import GenerateYP
# from .MyRegEncoder.Model import Encoder
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
#
# @MODELS.register_module()
# class LastPostLoss(nn.Module):
#     """MSELoss.
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
#         self.yp_generator = GenerateYP(class_num=1, img_height=128, img_width=128, device='cuda:0',
#                                        model_path='/home/jz207/workspace/liulb/MMDetection/Encoder_Kvasir-SEG_PreTrain7679.pth')
#         self.mask_resizer = MaskResizer(image_size=(1024, 1024))
#
#     def calculate_absolute_difference(self, student_output, coder_reg_output):
#         # 计算绝对差值并返回
#         return torch.abs(student_output - coder_reg_output)
#
#     def to_frequency_domain(self, absolute_difference):
#         # 将绝对差值转换到频域
#         frequency_domain = torch.fft.fft2(absolute_difference)
#         frequency_domain = torch.fft.fftshift(frequency_domain)
#         return frequency_domain
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
#         # 对学生特征进行对齐处理
#         x = align(s_input, t_input)
#
#         # 使用 GenerateYP 模块生成伪造标签 yp
#         yp = self.yp_generator.generate_yp(s_input)
#
#         # 计算与学生网络输出的绝对差值
#         absolute_difference = self.calculate_absolute_difference(x, yp)
#
#         # 转换学生特征到频域
#         s_out = self.to_frequency_domain(absolute_difference)
#
#         # 对教师特征进行 1x1 卷积和上采样
#         t_input = self.conv1x1(t_input)
#         t_input = self.upsample(t_input)
#
#         # 将教师特征转换到频域
#         fft_result = torch.fft.fft2(t_input)
#
#         # 创建一个高通滤波器掩码
#         _, c, h, w = t_input.shape
#         mask = torch.ones_like(fft_result)
#         cutoff_frequency_h = h // 4
#         cutoff_frequency_w = w // 4
#         mask[:, :, :cutoff_frequency_h, :cutoff_frequency_w] = 0
#         mask[:, :, -cutoff_frequency_h:, -cutoff_frequency_w:] = 0
#
#         # 应用掩码
#         fft_result *= mask
#         t_out = torch.fft.fftshift(fft_result)
#
#         # 取绝对值以计算 L1 损失
#         t_out = torch.abs(t_out)
#         s_out = torch.abs(s_out)
#
#         # 计算 L1 损失
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss = self.loss_weight * l1_loss(
#             s_out, t_out, weight, reduction=reduction, avg_factor=avg_factor)
#         return loss
