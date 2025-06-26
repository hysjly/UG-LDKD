# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, List, Union, Any,Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...registry import MODELS
from .utils import weighted_loss
from .MyRegEncoder.RegEncoderv2 import GenerateYPv1
from .MyRegEncoder.Model import Encoder
from .MyRegEncoder.Re import MaskResizer

@weighted_loss
def align(s: Tensor, t: Tensor) -> Tensor:
    """Align student feature map to match teacher feature map.
    Adjust channels and spatial size.

    Args:
        s (Tensor): Student mask tensor.
        t (Tensor): Teacher mask tensor.

    Returns:
        Tensor: Aligned student mask tensor.
    """
    s_channels = s.shape[1]
    t_channels = t.shape[1]
    t_height = t.shape[2]
    t_width = t.shape[3]

    if s_channels != t_channels or s.shape[2] != t_height or s.shape[3] != t_width:
        adjust_channels = nn.Sequential(
            nn.Conv2d(s_channels, t_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(t_channels),
            nn.ReLU(inplace=True)
        ).to(s.device)  # Ensure device consistency

        # Adjust channels
        s = adjust_channels(s)

        # Adjust spatial size if necessary
        if s.shape[2] != t_height or s.shape[3] != t_width:
            s = F.interpolate(s, size=(t_height, t_width), mode='bilinear', align_corners=True)

    return s



class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
        teacher_detach (bool): Whether to detach the teacher model prediction.
            Will set to ``'False'`` in some data-free distillation algorithms.
            Defaults to True.
    """

    def __init__(
        self,
        tau: float = 3.0,
        reduction: str = 'batchmean',
        loss_weight: float = 1.0,
        teacher_detach: bool = True,
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight
        self.teacher_detach = teacher_detach
        self.yp_generator = GenerateYPv1(class_num=1, img_height=256, img_width=256, device='cuda:0',
                                       model_path='/home/jz207/workspace/liull/MMDetection/Encoder_Kvasir_SEG_PreTrain_last.pth')
        self.mask_resizer = MaskResizer(image_size=(256, 256))

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction

    def forward(self,
                s_feat: Tensor,
                t_feat: Tensor,
                ):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        s_input, s_rois = s_feat
        t_input, t_rois = t_feat

        s_input = self.mask_resizer.convert_to_full_size_mask(s_input, s_rois)
        t_input = self.mask_resizer.convert_to_full_size_mask(t_input, t_rois)

        x = align(s_input, t_input)
        N, C, H, W = t_input.shape

        preds_S=s_input
        preds_T=t_input

        if self.teacher_detach:
            preds_T = preds_T.detach()
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
        return self.loss_weight * loss