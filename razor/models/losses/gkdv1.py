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

class ChannelWiseDivergenceWithU(_CWD):
    def __init__(self,
                 gamma=1.0,
                 epoch=0,
                 total_epochs=1000,
                 hd_epochs=1000,
                 consistency=1.0,
                 consistency_rampup=40.0,
                 hd_include_background=False,
                 one_hot_target=True,
                 softmax=True,
                 sigmoid: bool = False,
                 **kwargs):
        super(ChannelWiseDivergenceWithU, self).__init__(**kwargs)
        self.gamma = gamma
        self.epoch = epoch
        self.consistency = consistency
        self.consistency_rampup = consistency_rampup
        self.sigmoid = sigmoid
        self.criterion_kd = torch.nn.KLDivLoss()
        self.yp_generator = GenerateYPv1(class_num=13, img_height=256, img_width=256, device='cuda:0',
                                       model_path='/home/jz207/workspace/liull/MMDetection/Encoder_ChestXDet_PreTrain_last.pth')
        self.mask_resizer = MaskResizer(image_size=(256, 256)
                                        )

    def get_current_consistency_weight(self, epoch, consistency=0.1, consistency_rampup=40.0):
        return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

    def forward(self,
                s_feat: Tensor,
                t_feat: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward computation."""
        s_input, s_rois = s_feat
        t_input, t_rois = t_feat

        s_input = self.mask_resizer.convert_to_full_size_mask(s_input, s_rois)
        t_input = self.mask_resizer.convert_to_full_size_mask(t_input, t_rois)

        x = align(s_input, t_input)
        N, C, H, W = t_input.shape

        # Teacher masks
        yp_t = self.yp_generator.generate_yp(t_input)
        uncertainty = (F.softmax(t_input, dim=1) - F.softmax(yp_t, dim=1)) ** 2
        certainty = torch.exp(-1.0 * self.gamma * uncertainty)
        mask = certainty.float()

        # Student masks
        yp_s = self.yp_generator.generate_yp(s_input)
        uncertainty_stu = (F.softmax(s_input, dim=1) - F.softmax(yp_s, dim=1)) ** 2
        certainty_stu = torch.exp(self.gamma * uncertainty_stu)
        mask_stu = certainty_stu.float()

        # KL Loss
        softmax_pred_T = F.softmax(t_input.view(-1, H * W) / self.tau, dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        loss = torch.sum(softmax_pred_T *
                         logsoftmax(t_input.view(-1, H * W) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(s_input.view(-1, H * W) / self.tau)) * (self.tau ** 2)
        loss = self.loss_weight * loss / (C * N)

        # Uncertainty KD loss
        loss_u_kl = self.criterion_kd(
            F.log_softmax(uncertainty_stu, dim=1),
            F.softmax(uncertainty, dim=1))

        # # Combined loss
        # consistency_weight_u = self.get_current_consistency_weight(self.epoch, self.consistency, self.consistency_rampup)
        # total_loss = loss + consistency_weight_u * 10 * loss_u_kl

        loss_kl = self.criterion_kd(
            F.log_softmax(s_input / self.tau, dim=1),
            F.softmax(t_input / self.tau, dim=1)) * (self.tau ** 2)
        consistency_weight_u = self.get_current_consistency_weight(self.epoch, self.consistency,
                                                                           self.consistency_rampup)
        loss = torch.sum(mask * torch.exp(loss_kl) * loss) / (2 * torch.sum(mask) + 1e-16)
        total_loss = loss + consistency_weight_u * 10 * loss_u_kl
        # Return total loss and log components for monitoring
        self.log_components = dict(
            guide_loss=self.loss_weight * loss,
            noise_loss=consistency_weight_u * torch.log(2 * torch.sum(mask) + 1e-16),
            u_loss=self.loss_weight * consistency_weight_u * 10 * loss_u_kl
        )
        return total_loss
