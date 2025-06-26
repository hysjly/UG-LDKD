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


def euclidean_distance(pred, squared=False, eps=1e-12):
    """Calculate the Euclidean distance between the two examples in the output
    representation space."""
    pred_square = pred.pow(2).sum(dim=-1)  # (N, )
    prod = torch.mm(pred, pred.t())  # (N, N)
    distance = (pred_square.unsqueeze(1) + pred_square.unsqueeze(0) -
                2 * prod).clamp(min=eps)  # (N, N)

    if not squared:
        distance = distance.sqrt()

    distance = distance.clone()
    distance[range(len(prod)), range(len(prod))] = 0
    return distance


def angle(pred):
    """Calculate the angle-wise relational potential."""
    pred_vec = pred.unsqueeze(0) - pred.unsqueeze(1)  # (N, N, C)
    norm_pred_vec = F.normalize(pred_vec, p=2, dim=2)
    angle = torch.bmm(norm_pred_vec, norm_pred_vec.transpose(1, 2)).view(-1)  # (N*N*N, )
    return angle


def align(s, t):
    s_channels = s.shape[1]
    c = t.shape[1]
    h = t.shape[2]
    w = t.shape[3]
    adjust_channels = nn.Sequential(
        nn.Conv2d(s_channels, c, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True)).to("cuda")
    # 2. 调整学生特征的空间尺寸以匹配教师特征
    upsample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)

    s = adjust_channels(s)
    s = upsample(s)
    return s


class RelationalRKDLoss(nn.Module):
    """Relational Knowledge Distillation Loss combining distance-wise loss and
    angle-wise loss.

    <https://arxiv.org/abs/1904.05068>

    Args:
        distance_weight (float): Weight of distance-wise loss. Defaults to 25.0.
        angle_weight (float): Weight of angle-wise loss. Defaults to 50.0.
        with_l2_norm (bool): Whether to normalize the model predictions before
            calculating the loss. Defaults to True.
    """

    def __init__(self, distance_weight=25.0, angle_weight=50.0, with_l2_norm=True):
        super(RelationalRKDLoss, self).__init__()

        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.with_l2_norm = with_l2_norm
        self.yp_generator = GenerateYPv1(class_num=13, img_height=256, img_width=256, device='cuda:0',
                                       model_path='/home/jz207/workspace/liull/MMDetection/Encoder_ChestXDet_PreTrain_last.pth')
        self.mask_resizer = MaskResizer(image_size=(256, 256))

    def distance_loss(self, preds_S, preds_T):
        """Calculate distance-wise distillation loss."""
        d_T = euclidean_distance(preds_T, squared=False)
        mean_d_T = d_T[d_T > 0].mean()
        d_T = d_T / mean_d_T

        d_S = euclidean_distance(preds_S, squared=False)
        mean_d_S = d_S[d_S > 0].mean()
        d_S = d_S / mean_d_S

        return F.smooth_l1_loss(d_S, d_T)

    def angle_loss(self, preds_S, preds_T):
        """Calculate the angle-wise distillation loss."""
        angle_T = angle(preds_T)
        angle_S = angle(preds_S)
        return F.smooth_l1_loss(angle_S, angle_T)

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

        Returns:
            torch.Tensor: The calculated combined loss value.
        """
        s_input, s_rois = s_feat
        t_input, t_rois = t_feat

        s_input = self.mask_resizer.convert_to_full_size_mask(s_input, s_rois)
        t_input = self.mask_resizer.convert_to_full_size_mask(t_input, t_rois)

        x = align(s_input, t_input)
        N, C, H, W = t_input.shape

        preds_S=s_input
        preds_T=t_input

        if preds_S.shape != preds_T.shape:
            preds_S = align(preds_S, preds_T)

        # Flatten the predictions into (N, C)
        preds_S = preds_S.view(preds_S.shape[0], -1)
        preds_T = preds_T.view(preds_T.shape[0], -1)

        # Optionally normalize the features
        if self.with_l2_norm:
            preds_S = F.normalize(preds_S, p=2, dim=1)
            preds_T = F.normalize(preds_T, p=2, dim=1)

        # Calculate the two losses
        distance_loss = self.distance_loss(preds_S, preds_T)
        angle_loss = self.angle_loss(preds_S, preds_T)

        # Combine the losses
        total_loss = (self.distance_weight * distance_loss +
                      self.angle_weight * angle_loss)

        return total_loss