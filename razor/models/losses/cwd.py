# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import MODELS
from mmrazor.models.losses import ChannelWiseDivergence as _CWD
from razor.models.losses import ramps
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
class ChannelWiseDivergence(_CWD):

    def __init__(self, sigmoid: bool = False, **kwargs):
        super(ChannelWiseDivergence, self).__init__(**kwargs)
        self.sigmoid = sigmoid
        self.yp_generator = GenerateYPv1(class_num=13, img_height=256, img_width=256, device='cuda:0',
                                       model_path='/home/jz207/workspace/liull/MMDetection/Encoder_ChestXDet_PreTrain_last.pth')
        self.mask_resizer = MaskResizer(image_size=(256, 256))

    def forward(self,
                s_feat: Tensor,
                t_feat: Tensor,
                ):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        """Forward computation."""
        s_input, s_rois = s_feat
        t_input, t_rois = t_feat

        s_input = self.mask_resizer.convert_to_full_size_mask(s_input, s_rois)
        t_input = self.mask_resizer.convert_to_full_size_mask(t_input, t_rois)

        x = align(s_input, t_input)
        N, C, H, W = t_input.shape

        #  Loss
        softmax_pred_T = F.softmax(t_input.view(-1, H * W) / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(t_input.view(-1, H * W) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(s_input.view(-1, H * W) / self.tau)) * (self.tau ** 2)
        loss = self.loss_weight * loss / (C * N)

        return loss